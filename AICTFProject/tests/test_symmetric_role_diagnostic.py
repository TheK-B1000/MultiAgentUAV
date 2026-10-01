"""Symmetric role diagnostic (PI 2026-10-01): B gets the same role construction as A.

Covers the three new capabilities: a frozen non-contiguous seed list on an already-spent block, the
B-side ATTACK/DEFEND splice in the evaluator, and the trainer's spec-gated authorization of a non-A
defender."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from experiments import eval_specialist_crossover_scaled as E  # noqa: E402
from experiments import seed_registry as SR  # noqa: E402
from experiments import train_specialist_scale as T  # noqa: E402

REG, LO, HI = "T_BLOCK", 30200001, 30200010


def _registry(tmp_path, monkeypatch, status="SPENT"):
    monkeypatch.setattr(SR, "REGISTRY", tmp_path / "SEED_REGISTRY.json")
    SR.allocate(REG, LO, HI, "sealed_confirmatory", "primary", status=status)
    (tmp_path / "T_PRIMARY.json").write_text(json.dumps({"seeds": {"block": [LO, HI]}}), encoding="utf-8")


def _spec(seed_ids):
    return {"status": "FROZEN", "POST_HOC_MATCHED_ROLE_ABLATIONS": {"L": {
        "registry_experiment_id": REG, "block": f"{LO}..{HI}", "primary_record": "T_PRIMARY.json",
        "seed_ids": seed_ids}}}


# ------------------------------------------------------------------ frozen seed subsets
def test_frozen_subset_is_authorized_and_reports_the_block(tmp_path, monkeypatch):
    _registry(tmp_path, monkeypatch)
    out = E.post_hoc_block(_spec([LO + 1, LO + 7]), "L", REG, LO + 1, LO + 7, tmp_path, seeds=[LO + 7, LO + 1])
    assert out["block_range"] == (LO, HI)


@pytest.mark.parametrize("seeds,needle", [
    ([LO + 1], "exactly its frozen seed_ids"),            # a different list
    ([LO + 1, LO + 8], "exactly its frozen seed_ids"),
    (None, "exactly its frozen seed_ids"),                 # entry has a list, run gives none
])
def test_anything_but_the_frozen_list_refuses(tmp_path, monkeypatch, seeds, needle):
    _registry(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match=needle):
        E.post_hoc_block(_spec([LO + 1, LO + 7]), "L", REG, LO, HI, tmp_path, seeds=seeds)


def test_frozen_list_outside_its_block_refuses(tmp_path, monkeypatch):
    _registry(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="not inside its block"):
        E.post_hoc_block(_spec([LO + 1, HI + 5]), "L", REG, LO, HI, tmp_path, seeds=[LO + 1, HI + 5])


# ------------------------------------------------------------------ trainer authorization
def _args(tmp_path, **kw):
    ck = tmp_path / "pi_B.zip"
    ck.write_bytes(b"repaired pi_B")
    base = dict(team_size=2, role_k_defend=1, run_label_suffix="_sym_B", split_attack_defend_frozen_ckpt=str(ck),
                load_path=str(ck), symmetric_role_spec=str(tmp_path / "SPEC.json"))
    base.update(kw)
    return SimpleNamespace(**base), hashlib.sha256(b"repaired pi_B").hexdigest()


def _write_spec(tmp_path, sha, **over):
    entry = {"team_size": 2, "policy": "B", "role_k_defend": 1, "run_label_suffix": "_sym_B", "specialist_sha256": sha}
    entry.update(over)
    (tmp_path / "SPEC.json").write_text(json.dumps({"status": "FROZEN", "TRAINING_AUTHORIZED": [entry]}), encoding="utf-8")


def test_authorized_b_defender_passes(tmp_path):
    args, sha = _args(tmp_path)
    _write_spec(tmp_path, sha)
    T._require_symmetric_role_authorization(args, "B")


@pytest.mark.parametrize("kw,spec_over,needle", [
    ({"symmetric_role_spec": ""}, {}, "requires --symmetric-role-spec"),
    ({"role_k_defend": 2}, {}, "does not authorize"),
    ({"run_label_suffix": "_other"}, {}, "does not authorize"),
    ({}, {"specialist_sha256": "0" * 64}, "not the authorized repaired specialist"),
])
def test_unauthorized_b_defender_refuses(tmp_path, kw, spec_over, needle):
    args, sha = _args(tmp_path, **kw)
    _write_spec(tmp_path, sha, **spec_over)
    with pytest.raises(SystemExit, match=needle):
        T._require_symmetric_role_authorization(args, "B")


def test_defender_must_start_from_its_own_attacker(tmp_path):
    args, sha = _args(tmp_path)
    other = tmp_path / "other.zip"
    other.write_bytes(b"x")
    args.load_path = str(other)
    _write_spec(tmp_path, sha)
    with pytest.raises(SystemExit, match="warm-started from the same repaired specialist"):
        T._require_symmetric_role_authorization(args, "B")


# ------------------------------------------------------------------ end to end (real 2v2 episodes)
ROOT = Path(__file__).resolve().parents[1]
CK = ROOT / "artifacts/scale_2v2_specialists"
PI_DA = CK / "pi_A_specialist_2v2_std_split_defend_k1/ckpts/final_pi_A_specialist_2v2_std_split_defend_k1.zip"
PI_A = CK / "pi_A_specialist_2v2_std_entity_repair/ckpts/final_pi_A_specialist_2v2_std_entity_repair.zip"
PI_B = CK / "pi_B_specialist_2v2_std_entity_repair/ckpts/final_pi_B_specialist_2v2_std_entity_repair.zip"


@pytest.mark.skipif(not all(p.is_file() for p in (PI_DA, PI_A, PI_B)), reason="2v2 checkpoints not on disk")
def test_seed_list_with_both_sides_spliced(tmp_path, monkeypatch):
    """pi_DA stands in for a role-conditioned B defender: this exercises the B-side splice and roles
    plumbing on two non-contiguous seeds, not a scientific result."""
    _registry(tmp_path, monkeypatch)
    seeds = [LO + 2, LO + 6]
    spec = tmp_path / "SPEC.json"
    spec.write_text(json.dumps({**_spec(seeds), "confirmatory": False}), encoding="utf-8")
    (tmp_path / "seeds.json").write_text(json.dumps(seeds), encoding="utf-8")
    monkeypatch.setattr(E, "SD", tmp_path)
    monkeypatch.setattr(E, "ROOT", tmp_path)
    sha_a, sha_b = (hashlib.sha256(p.read_bytes()).hexdigest() for p in (PI_A, PI_B))
    monkeypatch.setattr(sys, "argv", [
        "eval", "--team-size", "2", "--spec", str(spec), "--post-hoc-ablation-spec", str(spec),
        "--pi-a-path", str(PI_DA), "--pi-b-path", str(PI_DA), "--seed-list", str(tmp_path / "seeds.json"),
        "--label", "L", "--device", "cpu", "--registry-experiment-id", REG, "--role-fixed-for-episode",
        "--role-k-defend", "1", "--frozen-attack-path", str(PI_A), "--frozen-attack-path-sha256", sha_a,
        "--frozen-attack-path-b", str(PI_B), "--frozen-attack-path-b-sha256", sha_b])
    E.main()
    rec = json.loads((tmp_path / "L_SPECIALIST_CROSSOVER_EVAL_RESULT.json").read_text(encoding="utf-8"))
    assert rec["status"] == "SEALED" and rec["arm"] == "POST_HOC_ABLATION" and rec["confirmatory"] is False
    assert rec["seed_ids"] == seeds and rec["split_policy_pi_B"]["frozen_attack_sha256"] == sha_b
    with (tmp_path / "l_specialist_crossover_eval_rows.csv").open(encoding="utf-8") as fh:
        assert sorted({int(r["seed"]) for r in csv.DictReader(fh)}) == seeds
    assert SR.load()["blocks"][0]["status"] == "SPENT"            # never changed
