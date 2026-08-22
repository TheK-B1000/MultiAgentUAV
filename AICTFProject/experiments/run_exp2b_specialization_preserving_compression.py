"""EXP2B specialization-preserving K=2 compression.

Scientific delta versus EXP2: live PPO cell assignment only.
EXP2 used 8/8/8/8 over z0|A,z0|B,z1|A,z1|B. EXP2B uses
16/0/0/16. Crossed cells remain terminal-evaluation conditions.

Default invocation is contract-only. ``--launch`` additionally requires the
committed implementation-gate artifact and an empty production directory.
"""
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

from experiments.opponent_spec import (  # noqa: E402
    assert_live_opponent_batch,
    install_keyed_opponent_overlays,
    pole_A_genome,
)
from experiments.run_exp2_k2_latent_compression import (  # noqa: E402
    TEACHER_HASHES,
    TEACHER_PATHS,
    build_exp2_config,
)
from rl.curriculum import phase_from_tag  # noqa: E402
from rl.training.orchestrator import orchestrate_training_run  # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand"
PROTOCOL = SD / "EXP2B_SPECIALIZATION_PRESERVING_LATENT_COMPRESSION_PROTOCOL.json"
PARENT_RESULT = SD / "EXP2_K2_LATENT_COMPRESSION_NOT_CONFIRMED.json"
IMPLEMENTATION_GATE = SD / "EXP2B_IMPLEMENTATION_GATE.json"
OUT = SD / "exp2b_specialization_preserving_compression"

TRAINING_SEED = 8_400_001
TOTAL_STEPS = 2_000_000
N_ENVS = 32
RUN_TAG = "exp2b_specialization_preserving_seed8400001_2m"
CELL_KEYS = ("OP6",) * 16 + ("OP7",) * 16
CELL_Z = (0,) * 16 + (1,) * 16
EXPECTED_CELLS = {"z0_A": 16, "z0_B": 0, "z1_A": 0, "z1_B": 16}

ALLOWED_CONFIG_DIFFS_VS_EXP2 = {
    "checkpoint_dir", "episode_csv_path", "exp2_protocol_path",
    "metrics_csv_path", "run_tag", "seed",
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
        key: {"EXP2": left.get(key), "EXP2B": right.get(key)}
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    }


def _load_protocol() -> dict[str, Any]:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != "EXP2B_SPECIALIZATION_PRESERVING_LATENT_COMPRESSION_V1":
        raise RuntimeError("wrong EXP2B protocol")
    if payload.get("status") != "FROZEN_BEFORE_IMPLEMENTATION_OR_TRAINING":
        raise RuntimeError("EXP2B protocol is not frozen pre-implementation")
    if payload["single_scientific_delta"]["EXP2B"] != EXPECTED_CELLS:
        raise RuntimeError("EXP2B protocol assignment drift")
    return payload


def build_exp2b_config():
    protocol = _load_protocol()
    cfg, exp2_contract = build_exp2_config()
    parent = dataclasses.asdict(cfg)
    art = OUT / RUN_TAG

    cfg.run_tag = RUN_TAG
    cfg.seed = TRAINING_SEED
    cfg.exp2_protocol_path = str(PROTOCOL)
    cfg.checkpoint_dir = str(art / "ckpts")
    cfg.metrics_csv_path = str(art / "metrics.csv")
    cfg.episode_csv_path = str(art / "episode_rows.csv")

    child = dataclasses.asdict(cfg)
    diff = _diff(parent, child)
    unexpected = sorted(set(diff) - ALLOWED_CONFIG_DIFFS_VS_EXP2)
    if unexpected:
        raise RuntimeError(f"EXP2B config drift outside identity/path axes: {unexpected}")
    if int(cfg.total_timesteps) != TOTAL_STEPS or int(cfg.n_envs) != N_ENVS:
        raise RuntimeError("EXP2B budget or environment count drift")
    if tuple(cfg.forced_latent_env_ids) != CELL_Z:
        raise RuntimeError("EXP2B persistent z assignment drift")

    return cfg, {
        "protocol_id": protocol["protocol_id"],
        "classification": "DIAGNOSTIC_SUPERVISED_COMPRESSION_CAUSAL_ABLATION",
        "parent": "EXP2_K2_LATENT_COMPRESSION_V1",
        "parent_resolved_config_sha256": _stable_hash(parent),
        "resolved_config_sha256": _stable_hash(child),
        "resolved_config_diff_vs_EXP2": diff,
        "allowed_config_diff_fields": sorted(ALLOWED_CONFIG_DIFFS_VS_EXP2),
        "single_scientific_delta_external_assignment": {
            "EXP2": {"z0_A": 8, "z0_B": 8, "z1_A": 8, "z1_B": 8},
            "EXP2B": EXPECTED_CELLS,
        },
        "teacher_sha256": list(TEACHER_HASHES),
        "seed": TRAINING_SEED,
        "total_environment_steps": TOTAL_STEPS,
        "terminal_checkpoint_only": True,
        "exp2_parent_contract": exp2_contract,
    }


def configure_exp2b_live_environment(
    env, cfg, *, contract: dict[str, Any], allow_development_seed: bool = False,
):
    seed_ok = 8_400_001 <= int(cfg.seed) <= 8_400_320
    if allow_development_seed:
        seed_ok = seed_ok or (8_500_001 <= int(cfg.seed) <= 8_500_192)
    if not seed_ok:
        raise RuntimeError(f"EXP2B seed escaped frozen block: {cfg.seed}")
    core = env.core
    if core.cfg.ruleset_id != "RULESET_V3_M1_OWN_FLAG_HOME":
        raise RuntimeError(f"EXP2B live ruleset drift: {core.cfg.ruleset_id!r}")
    if not bool(core.cfg.own_flag_home_required_to_score) or int(core.B) != N_ENVS:
        raise RuntimeError("EXP2B live M1 or 32-environment contract failed")

    core._bt_profile_override = None
    core._sds_opening_hold_steps = 0
    genomes = {"OP6": pole_A_genome()}
    install_keyed_opponent_overlays(core, genomes)
    for env_i, key in enumerate(CELL_KEYS):
        env.env_method("set_phase", phase_from_tag(key), indices=[env_i])
        env.env_method("set_next_opponent", "SCRIPTED", key, indices=[env_i])
    env.reset()
    rows = assert_live_opponent_batch(
        core, genomes, allowed_keys=("OP6", "OP7"),
        context="EXP2B specialization-preserving production construction",
    )
    realized = {
        "z0_A": sum(z == 0 and key == "OP6" for z, key in zip(CELL_Z, CELL_KEYS)),
        "z0_B": sum(z == 0 and key == "OP7" for z, key in zip(CELL_Z, CELL_KEYS)),
        "z1_A": sum(z == 1 and key == "OP6" for z, key in zip(CELL_Z, CELL_KEYS)),
        "z1_B": sum(z == 1 and key == "OP7" for z, key in zip(CELL_Z, CELL_KEYS)),
    }
    if realized != EXPECTED_CELLS:
        raise RuntimeError(f"EXP2B live cells are not 16/0/0/16: {realized}")
    for env_i, (z, row) in enumerate(zip(CELL_Z, rows)):
        expected_key = "OP6" if z == 0 else "OP7"
        if row["live_opponent_key"] != expected_key:
            raise RuntimeError(f"EXP2B z/pole mapping mismatch at env {env_i}")
    return {
        "exp2b_protocol": {
            **contract,
            "resolved_live_cells": realized,
            "resolved_opponent_rows": rows,
            "ruleset_id": core.cfg.ruleset_id,
            "own_flag_home_required_to_score": True,
        }
    }


def _preflight_launch() -> None:
    if not PARENT_RESULT.is_file() or json.loads(PARENT_RESULT.read_text(encoding="utf-8")).get("verdict") != "EXP2_K2_LATENT_COMPRESSION_NOT_CONFIRMED":
        raise RuntimeError("frozen EXP2 negative parent result is missing")
    if not IMPLEMENTATION_GATE.is_file():
        raise RuntimeError(f"EXP2B implementation gate missing: {IMPLEMENTATION_GATE}")
    gate = json.loads(IMPLEMENTATION_GATE.read_text(encoding="utf-8"))
    if gate.get("verdict") != "PASS" or gate.get("production_launch_authorized") is not True:
        raise RuntimeError("EXP2B implementation gate does not authorize launch")
    run_dir = OUT / RUN_TAG
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"EXP2B one-attempt directory is not empty: {run_dir}")
    for path, expected in zip(TEACHER_PATHS, TEACHER_HASHES):
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"EXP2B teacher hash mismatch: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    cfg, contract = build_exp2b_config()
    print(json.dumps({
        "mode": "LAUNCH" if args.launch else "CONTRACT_ONLY",
        "run_tag": cfg.run_tag,
        "seed": cfg.seed,
        "steps": cfg.total_timesteps,
        "cells": EXPECTED_CELLS,
        "teacher_sha256": contract["teacher_sha256"],
        "resolved_config_diff_vs_EXP2": contract["resolved_config_diff_vs_EXP2"],
        "scientific_delta": contract["single_scientific_delta_external_assignment"],
    }, indent=2, default=str))
    if not args.launch:
        print("CONTRACT ONLY. No environment constructed and no training step spent.")
        return 0
    _preflight_launch()
    orchestrate_training_run(
        cfg,
        pre_rollout_env_setup=partial(configure_exp2b_live_environment, contract=contract),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
