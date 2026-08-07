"""Durable Stage-1 / Stage-3 artifacts for C3 discovery (operational only).

Scientific cells remain owned by the frozen contract. These helpers only prevent
lost Stage-1 work and allow Stage-3 resume after process death.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


STAGE1_ANCHORS_NAME = "C3_STAGE1_ANCHORS.jsonl"
STAGE1_MANIFEST_NAME = "C3_STAGE1_MANIFEST.json"
STAGE3_RESULTS_NAME = "C3_STAGE3_ANCHOR_RESULTS.jsonl"
ABORT_RECORD_NAME = "C3_ABORTED_OPERATIONAL_SCALE.json"


def anchor_key(
    *,
    train_seed: int,
    opponent: str,
    eval_seed: int,
    pressure_step: int,
) -> str:
    return f"{int(train_seed)}|{opponent}|{int(eval_seed)}|{int(pressure_step)}"


def anchor_key_from_row(row: dict[str, Any]) -> str:
    return anchor_key(
        train_seed=int(row["train_seed"]),
        opponent=str(row["opponent"]),
        eval_seed=int(row["eval_seed"]),
        pressure_step=int(row["pressure_step"]),
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, default=str, allow_nan=False) + "\n")
        stream.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def write_stage1_artifacts(
    out_dir: Path,
    *,
    anchors: Iterable[dict[str, Any]],
    manifest: dict[str, Any],
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    anchors_path = out_dir / STAGE1_ANCHORS_NAME
    manifest_path = out_dir / STAGE1_MANIFEST_NAME
    with anchors_path.open("w", encoding="utf-8") as stream:
        for row in anchors:
            stream.write(json.dumps(row, default=str, allow_nan=False) + "\n")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    return anchors_path, manifest_path


def load_stage1_bundle(out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    anchors_path = out_dir / STAGE1_ANCHORS_NAME
    manifest_path = out_dir / STAGE1_MANIFEST_NAME
    if not anchors_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"Stage-1 artifacts missing under {out_dir} "
            f"(need {STAGE1_ANCHORS_NAME} and {STAGE1_MANIFEST_NAME})"
        )
    anchors = read_jsonl(anchors_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return anchors, manifest


def load_completed_stage3_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for row in read_jsonl(path):
        key = row.get("anchor_key")
        if key:
            keys.add(str(key))
    return keys
