"""Shared diagnostic seed block: several evaluation labels on the SAME seeds, declared up front.

The per-label default (one block per label) cannot express a paired multi-label diagnostic
without reusing seeds. The shared mode must refuse everything except exactly the declared
labels on exactly the registered, still-RESERVED range, and mark SPENT only when every declared
label has a sealed record.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from experiments import eval_specialist_crossover_scaled as E  # noqa: E402
from experiments import seed_registry as SR  # noqa: E402

LABELS = ["DIAG_PRESPLIT", "DIAG_SPLIT"]


@pytest.fixture
def reg(tmp_path, monkeypatch):
    monkeypatch.setattr(SR, "REGISTRY", tmp_path / "SEED_REGISTRY.json")
    SR.allocate("DIAG_SHARED", 20800001, 20800064, "exploratory", "paired diagnostic",
                shared_by_labels=LABELS)
    return tmp_path


def test_declared_label_on_exact_reserved_range_is_accepted(reg):
    for lab in LABELS:
        b = E.shared_block_owner("DIAG_SHARED", lab, 20800001, 20800064, "exploratory")
        assert b["shared_by_labels"] == LABELS


@pytest.mark.parametrize("reg_id,label,lo,hi,cls,needle", [
    ("NOPE", "DIAG_SPLIT", 20800001, 20800064, "exploratory", "not registered"),
    ("DIAG_SHARED", "DIAG_OTHER", 20800001, 20800064, "exploratory", "not declared"),
    ("DIAG_SHARED", "DIAG_SPLIT", 20800001, 20800128, "exploratory", "this run asks for"),
    ("DIAG_SHARED", "DIAG_SPLIT", 20800001, 20800064, "sealed_confirmatory", "the spec implies"),
])
def test_everything_else_is_refused(reg, reg_id, label, lo, hi, cls, needle):
    with pytest.raises(SystemExit, match=needle):
        E.shared_block_owner(reg_id, label, lo, hi, cls)


def test_spent_shared_block_is_refused(reg):
    SR.set_status("DIAG_SHARED", "SPENT")
    with pytest.raises(SystemExit, match="SPENT"):
        E.shared_block_owner("DIAG_SHARED", "DIAG_SPLIT", 20800001, 20800064, "exploratory")


def test_spent_only_once_every_declared_label_is_sealed(reg, tmp_path):
    b = next(x for x in SR.load()["blocks"] if x["experiment_id"] == "DIAG_SHARED")
    assert not E.shared_block_all_sealed(b, tmp_path)
    (tmp_path / "DIAG_PRESPLIT_SPECIALIST_CROSSOVER_EVAL_RESULT.json").write_text(
        json.dumps({"status": "SEALED"}), encoding="utf-8")
    assert not E.shared_block_all_sealed(b, tmp_path)
    (tmp_path / "DIAG_SPLIT_SPECIALIST_CROSSOVER_EVAL_RESULT.json").write_text(
        json.dumps({"status": "RUNNING"}), encoding="utf-8")
    assert not E.shared_block_all_sealed(b, tmp_path)          # not sealed yet
    (tmp_path / "DIAG_SPLIT_SPECIALIST_CROSSOVER_EVAL_RESULT.json").write_text(
        json.dumps({"status": "SEALED"}), encoding="utf-8")
    assert E.shared_block_all_sealed(b, tmp_path)


def test_a_normal_block_cannot_be_used_as_a_shared_one(reg):
    SR.allocate("PLAIN", 20900001, 20900064, "exploratory", "plain per-label block")
    with pytest.raises(SystemExit, match="not declared"):
        E.shared_block_owner("PLAIN", "DIAG_SPLIT", 20900001, 20900064, "exploratory")
