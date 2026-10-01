"""Symmetric-role top-50 seed lock: IDs frozen before any symmetric result."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
TOP = SD / "symmetric_role_top50"
SPEC = SD / "SYMMETRIC_ROLE_TOP50_DIAGNOSTIC_SPEC.json"
LOCK = TOP / "SEED_LOCK.json"
PROV = ROOT / "paper" / "aamas2027" / "true_top50_provenance"


@pytest.mark.parametrize("scale,prov_name,lo,hi", [
    ("2v2", "2v2_Ours_heuristic_roles_seeds.json", 23600001, 23600128),
    ("4v4", "4v4_Ours_heuristic_roles_seeds.json", 21800001, 21800128),
    ("6v6", "6v6_Ours_heuristic_roles_seeds.json", 25800001, 25800128),
])
def test_locked_ids_match_ours_top50_provenance(scale, prov_name, lo, hi):
    locked = [int(s) for s in json.loads((TOP / f"{scale}_ours_top50_seed_ids.json").read_text(encoding="utf-8"))]
    prov = sorted(int(s) for s in json.loads((PROV / prov_name).read_text(encoding="utf-8"))["selected_seed_ids"])
    assert locked == prov
    assert len(locked) == 50 and locked == sorted(set(locked))
    assert all(lo <= s <= hi for s in locked)


def test_spec_embeds_exact_locked_ids_and_forbids_128():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    assert str(spec["status"]).startswith("FROZEN")
    assert "STANDARDIZED_2V2_NOROLE_MATCHED" in str(spec["NOT_AUTHORIZED_BY_THIS_SPEC"])
    for label, entry in spec["POST_HOC_MATCHED_ROLE_ABLATIONS"].items():
        scale = "2v2" if "2V2" in label else "4v4" if "4V4" in label else "6v6"
        locked = [int(s) for s in json.loads((TOP / f"{scale}_ours_top50_seed_ids.json").read_text(encoding="utf-8"))]
        assert [int(s) for s in entry["seed_ids"]] == locked
        assert "n-seeds 128" not in json.dumps(entry)


def test_6v6_norole_covers_locked_seeds():
    ids = set(int(s) for s in json.loads((TOP / "6v6_ours_top50_seed_ids.json").read_text(encoding="utf-8")))
    rows_path = SD / "standardized_6v6_norole_specialist_crossover_eval_rows.csv"
    if not rows_path.is_file():
        pytest.skip("6v6 no-role rows not on disk")
    import csv
    with rows_path.open(encoding="utf-8") as fh:
        seeds = {int(r["seed"]) for r in csv.DictReader(fh)}
    assert ids <= seeds


def test_prepare_verify_lock_runs():
    from experiments.prepare_symmetric_role_top50 import verify_lock
    verify_lock()
