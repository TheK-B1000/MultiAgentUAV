"""Post-hoc matched role ablations (PI 2026-10-01): the evaluator may reuse a confirmatory block only
when a frozen spec lists the exact label/block, the block is SPENT, and the matched primary record
was sealed on exactly that block. It never spends fresh seeds."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from experiments import eval_specialist_crossover_scaled as E  # noqa: E402
from experiments import seed_registry as SR  # noqa: E402

LABEL, REG, LO, HI = "T_NOROLE_MATCHED", "T_CONFIRMATORY_SPECIALIST_CROSSOVER", 30100001, 30100004


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(SR, "REGISTRY", tmp_path / "SEED_REGISTRY.json")
    SR.allocate(REG, LO, HI, "sealed_confirmatory", "primary confirmatory", status="SPENT")
    (tmp_path / "T_PRIMARY_RESULT.json").write_text(json.dumps({"seeds": {"block": [LO, HI]}}), encoding="utf-8")
    spec = {"status": "FROZEN", "POST_HOC_MATCHED_ROLE_ABLATIONS": {
        LABEL: {"registry_experiment_id": REG, "block": f"{LO}..{HI}", "primary_record": "T_PRIMARY_RESULT.json"}}}
    return tmp_path, spec


def test_exact_spent_block_is_authorized(env):
    sd, spec = env
    out = E.post_hoc_block(spec, LABEL, REG, LO, HI, sd)
    assert out["seed_class"] == "sealed_confirmatory"
    assert SR.load()["blocks"][0]["status"] == "SPENT"              # unchanged


@pytest.mark.parametrize("label,reg,lo,hi,needle", [
    ("OTHER_LABEL", REG, LO, HI, "not a post-hoc ablation"),
    (LABEL, "SOME_OTHER_BLOCK", LO, HI, "is authorized on"),
    (LABEL, REG, LO, HI + 1, "is authorized on"),
])
def test_anything_not_listed_refuses(env, label, reg, lo, hi, needle):
    sd, spec = env
    with pytest.raises(SystemExit, match=needle):
        E.post_hoc_block(spec, label, reg, lo, hi, sd)


def test_reserved_block_cannot_be_spent_through_this_path(tmp_path, monkeypatch):
    monkeypatch.setattr(SR, "REGISTRY", tmp_path / "SEED_REGISTRY.json")
    SR.allocate(REG, LO, HI, "sealed_confirmatory", "still reserved")
    (tmp_path / "T_PRIMARY_RESULT.json").write_text(json.dumps({"seeds": {"block": [LO, HI]}}), encoding="utf-8")
    spec = {"POST_HOC_MATCHED_ROLE_ABLATIONS": {LABEL: {"registry_experiment_id": REG, "block": f"{LO}..{HI}",
                                                        "primary_record": "T_PRIMARY_RESULT.json"}}}
    with pytest.raises(SystemExit, match="requires a SPENT block"):
        E.post_hoc_block(spec, LABEL, REG, LO, HI, tmp_path)


def test_primary_record_on_other_seeds_refuses(env):
    sd, spec = env
    (sd / "T_PRIMARY_RESULT.json").write_text(json.dumps({"seeds": {"block": [LO + 100, HI + 100]}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="was sealed on"):
        E.post_hoc_block(spec, LABEL, REG, LO, HI, sd)
