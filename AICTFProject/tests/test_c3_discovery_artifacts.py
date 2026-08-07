from __future__ import annotations

import json
from pathlib import Path

from rl.analysis.c3_discovery_artifacts import (
    STAGE1_ANCHORS_NAME,
    STAGE1_MANIFEST_NAME,
    STAGE3_RESULTS_NAME,
    anchor_key,
    anchor_key_from_row,
    append_jsonl,
    load_completed_stage3_keys,
    load_stage1_bundle,
    write_stage1_artifacts,
)


def test_stage1_persist_and_reload_roundtrip(tmp_path: Path):
    anchors = [
        {
            "train_seed": 3200001,
            "opponent": "OP6",
            "eval_seed": 9400000,
            "pressure_step": 12,
            "c3_contract_hash": "abc",
            "checkpoint_sha256": "def",
            "map": "map_a",
            "ruleset": "RULESET_V2_AQUATICUS_10S",
        }
    ]
    manifest = {
        "status": "STAGE1_FROZEN",
        "c3_contract_hash": "abc",
        "n_anchors": 1,
    }
    write_stage1_artifacts(tmp_path, anchors=anchors, manifest=manifest)
    assert (tmp_path / STAGE1_ANCHORS_NAME).exists()
    assert (tmp_path / STAGE1_MANIFEST_NAME).exists()
    loaded, loaded_manifest = load_stage1_bundle(tmp_path)
    assert loaded == anchors
    assert loaded_manifest["c3_contract_hash"] == "abc"


def test_stage3_resume_skips_completed_anchor_keys(tmp_path: Path):
    path = tmp_path / STAGE3_RESULTS_NAME
    row = {
        "anchor_key": anchor_key(
            train_seed=3200001,
            opponent="OP6",
            eval_seed=9400001,
            pressure_step=44,
        ),
        "train_seed": 3200001,
        "opponent": "OP6",
        "eval_seed": 9400001,
        "pressure_step": 44,
        "episode_status": "NO_COMMITMENT_FORK",
    }
    append_jsonl(path, row)
    append_jsonl(
        path,
        {
            "anchor_key": anchor_key(
                train_seed=3200001,
                opponent="OP6",
                eval_seed=9400002,
                pressure_step=10,
            ),
            "episode_status": "QUALIFIED_COMMITMENT_FORK",
        },
    )
    keys = load_completed_stage3_keys(path)
    assert len(keys) == 2
    assert anchor_key_from_row(row) in keys
    pending = [
        {"train_seed": 3200001, "opponent": "OP6", "eval_seed": 9400001, "pressure_step": 44},
        {"train_seed": 3200001, "opponent": "OP6", "eval_seed": 9400003, "pressure_step": 7},
    ]
    remaining = [a for a in pending if anchor_key_from_row(a) not in keys]
    assert len(remaining) == 1
    assert remaining[0]["eval_seed"] == 9400003


def test_abort_record_schema_is_operational_not_scientific(tmp_path: Path):
    payload = {
        "status": "ABORTED_OPERATIONAL_SCALE",
        "scientific_verdict": "NO_SCIENTIFIC_VERDICT",
        "pid": 12820,
        "notes": "Killed due to Stage-3 combinatorial wall-clock; does not count against C3.",
    }
    path = tmp_path / "C3_ABORTED_OPERATIONAL_SCALE.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["status"] == "ABORTED_OPERATIONAL_SCALE"
    assert loaded["scientific_verdict"] == "NO_SCIENTIFIC_VERDICT"
