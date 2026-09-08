"""EXP2C K=2 compression with z-specific final actor projections only."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_exp2_k2_latent_compression import TEACHER_HASHES, TEACHER_PATHS
from experiments.run_exp2b_specialization_preserving_compression import (
    EXPECTED_CELLS,
    build_exp2b_config,
    configure_exp2b_live_environment,
)
from rl.training.orchestrator import orchestrate_training_run

SD = ROOT / "artifacts" / "strategic_demand"
PROTOCOL = SD / "EXP2C_MODE_SPECIFIC_ACTOR_COMPRESSION_PROTOCOL.json"
PARENT_RESULT = SD / "EXP2B_SPECIALIZATION_PRESERVING_LATENT_COMPRESSION_NOT_CONFIRMED.json"
IMPLEMENTATION_GATE = SD / "EXP2C_IMPLEMENTATION_GATE.json"
OUT = SD / "exp2c_mode_specific_actor_compression"

TRAINING_SEED = 8_700_001
TOTAL_STEPS = 2_000_000
RUN_TAG = "exp2c_mode_specific_actor_seed8700001_2m"
ALLOWED_CONFIG_DIFFS_VS_EXP2B = {
    "checkpoint_dir", "episode_csv_path", "exp2_protocol_path",
    "exp2c_mode_specific_action_heads", "metrics_csv_path", "run_tag", "seed",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _diff(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        key: {"EXP2B": left.get(key), "EXP2C": right.get(key)}
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    }


def _load_protocol() -> dict[str, Any]:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != "EXP2C_MODE_SPECIFIC_ACTOR_COMPRESSION_V1":
        raise RuntimeError("wrong EXP2C protocol")
    if payload.get("status") != "FROZEN_BEFORE_IMPLEMENTATION_OR_TRAINING":
        raise RuntimeError("EXP2C protocol is not frozen before implementation/training")
    return payload


def build_exp2c_config():
    protocol = _load_protocol()
    cfg, parent_contract = build_exp2b_config()
    parent = dataclasses.asdict(cfg)
    art = OUT / RUN_TAG
    cfg.run_tag = RUN_TAG
    cfg.seed = TRAINING_SEED
    cfg.exp2_protocol_path = str(PROTOCOL)
    cfg.exp2c_mode_specific_action_heads = True
    cfg.checkpoint_dir = str(art / "ckpts")
    cfg.metrics_csv_path = str(art / "metrics.csv")
    cfg.episode_csv_path = str(art / "episode_rows.csv")
    child = dataclasses.asdict(cfg)
    diff = _diff(parent, child)
    unexpected = sorted(set(diff) - ALLOWED_CONFIG_DIFFS_VS_EXP2B)
    if unexpected:
        raise RuntimeError(f"EXP2C config drift outside frozen axes: {unexpected}")
    scientific = sorted(set(diff) - {
        "checkpoint_dir", "episode_csv_path", "exp2_protocol_path",
        "metrics_csv_path", "run_tag", "seed",
    })
    if scientific != ["exp2c_mode_specific_action_heads"]:
        raise RuntimeError(f"EXP2C scientific diff is not single-axis: {scientific}")
    if int(cfg.total_timesteps) != TOTAL_STEPS:
        raise RuntimeError("EXP2C budget drift")
    return cfg, {
        "protocol_id": protocol["protocol_id"],
        "classification": protocol["classification"],
        "parent": "EXP2B_SPECIALIZATION_PRESERVING_LATENT_COMPRESSION_V1",
        "parent_resolved_config_sha256": _stable_hash(parent),
        "resolved_config_sha256": _stable_hash(child),
        "resolved_config_diff_vs_EXP2B": diff,
        "scientific_diff_fields": scientific,
        "cells": EXPECTED_CELLS,
        "teacher_sha256": list(TEACHER_HASHES),
        "seed": TRAINING_SEED,
        "total_environment_steps": TOTAL_STEPS,
        "terminal_checkpoint_only": True,
        "parent_contract": parent_contract,
    }


def configure_exp2c_live_environment(env, cfg, *, contract, allow_development_seed=False):
    return configure_exp2b_live_environment(
        env,
        cfg,
        contract=contract,
        allow_development_seed=allow_development_seed,
        training_seed_range=(8_700_001, 8_700_320),
        development_seed_range=(8_800_001, 8_800_192),
        manifest_key="exp2c_protocol",
        context_label="EXP2C mode-specific actor production construction",
    )


def _preflight_launch() -> None:
    parent = json.loads(PARENT_RESULT.read_text(encoding="utf-8"))
    if parent.get("verdict") != "EXP2B_SPECIALIZATION_PRESERVING_LATENT_COMPRESSION_NOT_CONFIRMED":
        raise RuntimeError("frozen EXP2B negative parent result is missing")
    gate = json.loads(IMPLEMENTATION_GATE.read_text(encoding="utf-8"))
    if gate.get("verdict") != "PASS" or gate.get("production_launch_authorized") is not True:
        raise RuntimeError("EXP2C implementation gate does not authorize launch")
    run_dir = OUT / RUN_TAG
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"EXP2C one-attempt directory is not empty: {run_dir}")
    for path, expected in zip(TEACHER_PATHS, TEACHER_HASHES):
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"EXP2C teacher hash mismatch: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    cfg, contract = build_exp2c_config()
    print(json.dumps({
        "mode": "LAUNCH" if args.launch else "CONTRACT_ONLY",
        "run_tag": cfg.run_tag,
        "seed": cfg.seed,
        "steps": cfg.total_timesteps,
        "cells": EXPECTED_CELLS,
        "teacher_sha256": contract["teacher_sha256"],
        "resolved_config_diff_vs_EXP2B": contract["resolved_config_diff_vs_EXP2B"],
    }, indent=2, default=str))
    if not args.launch:
        print("CONTRACT ONLY. No environment constructed and no training step spent.")
        return 0
    _preflight_launch()
    orchestrate_training_run(
        cfg,
        pre_rollout_env_setup=partial(configure_exp2c_live_environment, contract=contract),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
