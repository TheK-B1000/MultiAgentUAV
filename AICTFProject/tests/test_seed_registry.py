"""Rule 9 self-test: the registry must REFUSE, not merely record.

A registry that lists blocks but cannot reject a re-spend is documentation, not
a control. Every test here is a refusal the launcher depends on.
"""

from __future__ import annotations

import json

import pytest

from experiments import seed_registry as SR


@pytest.fixture
def reg(tmp_path, monkeypatch):
    monkeypatch.setattr(SR, "REGISTRY", tmp_path / "SEED_REGISTRY.json")
    SR.allocate("CONF_A", 20000001, 20000128, "sealed_confirmatory",
                "sealed crossover", status="SPENT")
    SR.allocate("EXPL_A", 20100001, 20100032, "exploratory", "a diagnostic")
    return tmp_path


def test_exact_respend_of_confirmatory_block_is_refused(reg):
    ok, msg = SR.check_block(20000001, 20000128, "sealed_confirmatory")
    assert not ok and "OVERLAPS" in msg and "CONF_A" in msg


def test_partial_overlap_is_refused(reg):
    """The dangerous case: a block that looks new but shares its tail."""
    ok, msg = SR.check_block(20000100, 20000200, "exploratory")
    assert not ok and "OVERLAPS" in msg


@pytest.mark.parametrize("lo,hi", [(20000001, 20000128), (19999999, 20000002),
                                   (20000127, 20000300), (20000064, 20000064)])
def test_every_flavour_of_collision_is_refused(reg, lo, hi):
    assert not SR.check_block(lo, hi, "exploratory")[0]


def test_free_block_is_allowed(reg):
    ok, msg = SR.check_block(20200001, 20200064, "exploratory")
    assert ok and "free" in msg


def test_allocate_refuses_on_overlap(reg):
    with pytest.raises(SystemExit, match="REFUSING to allocate"):
        SR.allocate("NEW", 20000050, 20000060, "exploratory", "oops")


def test_allocate_refuses_duplicate_experiment_id(reg):
    with pytest.raises(SystemExit, match="already registered"):
        SR.allocate("CONF_A", 20300001, 20300008, "exploratory", "dup id")


def test_smoke_family_is_reserved_in_both_directions(reg):
    assert not SR.check_block(20400001, 20400008, "smoke")[0]
    assert not SR.check_block(SR.SMOKE_LO + 1, SR.SMOKE_LO + 8,
                              "sealed_confirmatory")[0]
    assert SR.check_block(SR.SMOKE_LO + 1, SR.SMOKE_LO + 8, "smoke")[0]


def test_spent_confirmatory_cannot_be_reopened(reg):
    """The whole point of the class: confirmatory seeds are spent exactly once."""
    with pytest.raises(SystemExit, match="spent exactly once"):
        SR.set_status("CONF_A", "RESERVED")
    SR.set_status("CONF_A", "RETIRED", note="superseded")     # the one legal move
    assert SR.load()["blocks"][0]["status"] == "RETIRED"


def test_declared_subdivision_is_allowed_but_undeclared_nesting_is_not(reg):
    """A 320-seed collection bank split into shards is legitimate; a block that
    silently sits inside another is not."""
    SR.allocate("BANK", 20500001, 20500320, "exploratory", "collection bank")
    ok, _ = SR.check_block(20500001, 20500080, "exploratory", subdivides="BANK")
    assert ok
    assert not SR.check_block(20500001, 20500080, "exploratory")[0]
    with pytest.raises(SystemExit, match="not contained in parent"):
        SR.allocate("SHARD_BAD", 20500300, 20500400, "exploratory", "spills out",
                    subdivides="BANK")


def test_registry_audit_detects_hand_edited_overlap(reg):
    """Someone editing the JSON by hand must not be able to sneak past."""
    assert SR.audit_registry()[0]
    doc = SR.load()
    doc["blocks"].append({"experiment_id": "SNEAKY", "lo": 20000010, "hi": 20000020,
                          "n": 11, "seed_class": "exploratory", "status": "SPENT",
                          "subdivides": None, "purpose": "hand-edited"})
    SR.save(doc)
    ok, problems = SR.audit_registry()
    assert not ok and any("UNDECLARED OVERLAP" in p for p in problems)


def test_next_free_never_returns_a_colliding_block(reg):
    for n in (8, 32, 128, 320):
        lo = SR.next_free(n)
        assert SR.check_block(lo, lo + n - 1, "exploratory")[0]


def test_unknown_class_is_flagged_not_silently_accepted(reg):
    assert not SR.check_block(20600001, 20600008, "confirmatory")[0]   # typo'd class
    doc = SR.load()
    doc["blocks"].append({"experiment_id": "U", "lo": 20700001, "hi": 20700008, "n": 8,
                          "seed_class": "UNCLASSIFIED", "status": "SPENT",
                          "subdivides": None, "purpose": "backfilled, unresolved"})
    SR.save(doc)
    ok, problems = SR.audit_registry()
    assert not ok and any("do not guess" in p for p in problems)


# ------------------------------------------- integration with Rule 7 sealing --
def test_seal_audit_gates_on_the_seed_registry(reg, tmp_path, monkeypatch):
    """An evaluator declaring a seed class must be checked against the registry,
    so a confirmatory re-spend is caught at seal time even if the launcher was
    bypassed."""
    import csv

    from experiments.run_state import AuditPlan, RunState, seal

    seeds = list(range(20000001, 20000009))          # inside SPENT confirmatory CONF_A
    rows = tmp_path / "r.csv"
    with rows.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["pole", "seed", "win"])
        w.writeheader()
        for s in seeds:
            w.writerow({"pole": "A", "seed": s, "win": 1})

    plan = AuditPlan(rows_csv=rows, expected_rows=len(seeds), expected_seeds=seeds,
                     group_by=["pole"], seed_class="sealed_confirmatory", n_boot=200)
    st = RunState(tmp_path, "SEEDGATE").begin()
    audit = seal(out_path=tmp_path / "SEEDGATE_RESULT.json", payload={"r": 1},
                 plan=plan, state=st, strict=False)
    assert not audit["passed"]
    assert "seed_class" in audit["failed_checks"]
    assert st.state == "AUDIT_FAILED"


# --------------------------------------- five-way self-check decision table --
def test_decision_table_free_block_is_allowed(reg):
    """A block nothing has ever touched: allow."""
    ok, _ = SR.check_block(20800001, 20800032, "exploratory", experiment_id="ANY_ID")
    assert ok


def test_decision_table_block_allocated_to_another_experiment_is_rejected(reg):
    ok, msg = SR.check_block(20000001, 20000128, "sealed_confirmatory",
                             experiment_id="SOMEONE_ELSE")
    assert not ok and "CONF_A" in msg


def test_decision_table_block_allocated_to_this_experiment_is_allowed(reg):
    """The self-check case the CLI fix exists for."""
    ok, msg = SR.check_block(20000001, 20000128, "sealed_confirmatory",
                             experiment_id="CONF_A")
    assert ok and "CONF_A" in msg


def test_decision_table_overlapping_historical_block_is_rejected_regardless_of_status(reg):
    SR.set_status("CONF_A", "RETIRED")
    ok, msg = SR.check_block(20000050, 20000060, "exploratory")
    assert not ok and "CONF_A" in msg


def test_decision_table_mismatched_range_under_own_id_is_rejected(reg):
    """THE BUG: claiming your own experiment_id but a DIFFERENT range that
    overlaps your real reservation must still be rejected, not waved through."""
    ok, msg = SR.check_block(20000001, 20000200, "sealed_confirmatory",
                             experiment_id="CONF_A")          # real block ends at 20000128
    assert not ok and "OVERLAPS" in msg


def test_decision_table_own_id_disjoint_from_own_prior_block_is_a_fresh_check(reg):
    """Same experiment_id, a range that does NOT overlap the existing entry at
    all: not an exact match, but also nothing to clash with -- evaluated as a
    normal (here: free) request, not auto-approved and not auto-rejected."""
    ok, msg = SR.check_block(20900001, 20900032, "exploratory", experiment_id="CONF_A")
    assert ok and "free" in msg
