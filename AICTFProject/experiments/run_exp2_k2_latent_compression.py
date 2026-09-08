"""EXP2 K=2 supervised latent repertoire compression.

Proposed Preset Review
----------------------
Name: EXP2_K2_LATENT_COMPRESSION_V1
Scientific question: Can one shared policy conditioned on a persistent binary
latent preserve the already-confirmed SAPPO two-policy repertoire?
Parent: experiments.run_g0_v5_long.build_config(seed=3_200_001)
Classification: DIAGNOSTIC_SUPERVISED_COMPRESSION
Actor changed: yes, one shared concat-conditioned K=2 actor.
Router/q_phi: absent. Latents are externally assigned, not discovered.
Reward: unchanged. Opponent identity input: absent from actor, critic, and z.
Supervision: online KL(pi_SAPPO teacher || pi_student(.|o,z)).
Resampling: none; each vector slot has one persistent z for every episode.

This is supervised latent repertoire compression, not unsupervised or
label-free latent strategy discovery.

The default command is contract-only. Production training requires both the
explicit ``--launch`` flag and a passing implementation-gate artifact.
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
from experiments.run_g0_v5_long import build_config as build_g0_v5_config  # noqa: E402
from rl.curriculum import phase_from_tag  # noqa: E402
from rl.training.orchestrator import orchestrate_training_run  # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand"
PROTOCOL = SD / "EXP2_K2_LATENT_COMPRESSION_PROTOCOL.json"
IMPLEMENTATION_GATE = SD / "EXP2_K2_IMPLEMENTATION_GATE.json"
OUT = SD / "exp2_k2_latent_compression"

PARENT_SEED = 3_200_001
TRAINING_SEED = 8_100_001
TOTAL_STEPS = 2_000_000
N_ENVS = 32
RUN_TAG = "exp2_k2_supervised_compression_seed8100001_2m"
TEACHER_PATHS = (
    SD / "sappo_continuation/sappo_pi_A_specialist_1p5M_seed7100001/ckpts/final_sappo_pi_A_specialist_1p5M_seed7100001.zip",
    SD / "sappo_continuation/sappo_pi_B_specialist_1p5M_seed7200001/ckpts/final_sappo_pi_B_specialist_1p5M_seed7200001.zip",
)
TEACHER_HASHES = (
    "5bd5f54f5ce206b139626bded8ca1f296d82d47c0d4c21db4ed561297a2d411d",
    "8e4fb58be11465c24a258da3ac94648e669c0f65ab98a64b42a7b4c8b6a6c8fc",
)

# Slots 0..7 z0/A, 8..15 z0/B, 16..23 z1/A, 24..31 z1/B.
CELL_KEYS = ("OP6",) * 8 + ("OP7",) * 8 + ("OP6",) * 8 + ("OP7",) * 8
CELL_Z = (0,) * 16 + (1,) * 16

ALLOWED_DIFFS = {
    "checkpoint_dir", "episode_csv_path", "metrics_csv_path", "run_tag",
    "seed", "total_timesteps", "periodic_checkpoint_steps", "n_envs",
    "own_flag_home_required_to_score", "fixed_opponent_tag", "opponent_pool",
    "opponent_pool_weights", "use_latent_strategy", "latent_k",
    "latent_strategy_encoder_enabled", "latent_assignment_mode",
    "forced_latent_env_ids", "latent_entropy_objective", "latent_lam_h",
    "latent_lam_p", "latent_strategy_ppo_coef",
    "exp2_teacher_compression_enabled", "exp2_teacher_checkpoints",
    "exp2_teacher_sha256", "exp2_teacher_lambda", "exp2_teacher_cadence",
    "exp2_teacher_batch_size", "exp2_protocol_path",
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


def _config_diff(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    return {
        key: {"g0_v5": parent.get(key), "exp2": child.get(key)}
        for key in sorted(set(parent) | set(child))
        if parent.get(key) != child.get(key)
    }


def _load_frozen_protocol() -> dict[str, Any]:
    if not PROTOCOL.is_file():
        raise RuntimeError(f"frozen EXP2 protocol is missing: {PROTOCOL}")
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != "EXP2_K2_LATENT_COMPRESSION_V1":
        raise RuntimeError("wrong EXP2 protocol_id")
    if payload.get("status") != "FROZEN_BEFORE_IMPLEMENTATION_OR_TRAINING":
        raise RuntimeError("EXP2 protocol is not frozen in its pretraining state")
    if payload["training_distribution"]["static_equal_cells"] != {
        "z0_A": 8, "z0_B": 8, "z1_A": 8, "z1_B": 8,
    }:
        raise RuntimeError("frozen protocol cell allocation drifted")
    return payload


def build_exp2_config():
    """Return the frozen EXP2 config and a field-by-field parent diff."""
    protocol = _load_frozen_protocol()
    cfg = build_g0_v5_config(PARENT_SEED)
    parent = dataclasses.asdict(cfg)
    art = OUT / RUN_TAG

    cfg.run_tag = RUN_TAG
    cfg.seed = TRAINING_SEED
    cfg.total_timesteps = TOTAL_STEPS
    cfg.periodic_checkpoint_steps = 100_000
    cfg.own_flag_home_required_to_score = True
    cfg.fixed_opponent_tag = "OP6"
    cfg.opponent_pool = ("OP6", "OP7")
    cfg.opponent_pool_weights = (0.5, 0.5)
    cfg.n_envs = N_ENVS

    cfg.use_latent_strategy = True
    cfg.latent_k = 2
    cfg.latent_strategy_encoder_enabled = False
    cfg.latent_assignment_mode = "static_env"
    cfg.forced_latent_env_ids = CELL_Z
    cfg.latent_entropy_objective = "none"
    cfg.latent_lam_h = 0.0
    cfg.latent_lam_p = 0.0
    cfg.latent_strategy_ppo_coef = 0.0

    cfg.exp2_teacher_compression_enabled = True
    cfg.exp2_teacher_checkpoints = tuple(str(path) for path in TEACHER_PATHS)
    cfg.exp2_teacher_sha256 = TEACHER_HASHES
    cfg.exp2_teacher_lambda = 0.10
    cfg.exp2_teacher_cadence = 4
    cfg.exp2_teacher_batch_size = 64
    cfg.exp2_protocol_path = str(PROTOCOL)

    cfg.load_path = None
    cfg.additional_timesteps = 0
    cfg.load_weights_only = False
    cfg.checkpoint_dir = str(art / "ckpts")
    cfg.metrics_csv_path = str(art / "metrics.csv")
    cfg.episode_csv_path = str(art / "episode_rows.csv")

    child = dataclasses.asdict(cfg)
    diff = _config_diff(parent, child)
    unexpected = sorted(set(diff) - ALLOWED_DIFFS)
    if unexpected:
        raise RuntimeError(f"EXP2 config drift outside frozen axes: {unexpected}")
    if int(cfg.n_envs) != 32 or len(CELL_KEYS) != 32 or len(CELL_Z) != 32:
        raise RuntimeError("EXP2 requires exactly 32 vector environments")

    contract = {
        "protocol_id": protocol["protocol_id"],
        "classification": "DIAGNOSTIC_SUPERVISED_COMPRESSION",
        "claim_boundary": (
            "Supervised latent repertoire compression, not unsupervised or "
            "label-free latent strategy discovery."
        ),
        "parent": "experiments.run_g0_v5_long.build_config",
        "parent_resolved_config_sha256": _stable_hash(parent),
        "resolved_config_sha256": _stable_hash(child),
        "resolved_config_diff": diff,
        "allowed_diff_fields": sorted(ALLOWED_DIFFS),
        "teacher_mapping": {"z0": "pi_A_SAPPO", "z1": "pi_B_SAPPO"},
        "teacher_sha256": list(TEACHER_HASHES),
        "static_cells": {"z0_A": 8, "z0_B": 8, "z1_A": 8, "z1_B": 8},
        "seed": TRAINING_SEED,
        "total_environment_steps": TOTAL_STEPS,
        "terminal_checkpoint_only": True,
    }
    return cfg, contract


def configure_exp2_live_environment(env, cfg, *, contract: dict[str, Any]):
    """Install and prove the immutable 8/8/8/8 pole/latent batch."""
    if int(cfg.seed) != TRAINING_SEED or not (8_100_001 <= int(cfg.seed) <= 8_100_320):
        raise RuntimeError(f"EXP2 training seed escaped frozen block: {cfg.seed}")
    if int(cfg.total_timesteps) != TOTAL_STEPS:
        raise RuntimeError(f"EXP2 budget drifted: {cfg.total_timesteps}")
    core = env.core
    if core.cfg.ruleset_id != "RULESET_V3_M1_OWN_FLAG_HOME":
        raise RuntimeError(f"EXP2 live ruleset is {core.cfg.ruleset_id!r}, expected M1")
    if not bool(core.cfg.own_flag_home_required_to_score):
        raise RuntimeError("EXP2 M1 own-flag-home rule is disabled")
    if int(core.B) != N_ENVS:
        raise RuntimeError(f"EXP2 live batch has {int(core.B)} envs, expected 32")

    core._bt_profile_override = None
    core._sds_opening_hold_steps = 0
    genomes = {"OP6": pole_A_genome()}
    install_keyed_opponent_overlays(core, genomes)
    for env_i, key in enumerate(CELL_KEYS):
        env.env_method("set_phase", phase_from_tag(key), indices=[env_i])
        env.env_method("set_next_opponent", "SCRIPTED", key, indices=[env_i])
    env.reset()
    rows = assert_live_opponent_batch(
        core,
        genomes,
        allowed_keys=("OP6", "OP7"),
        context="EXP2 K=2 production construction",
    )
    realized = {
        "z0_A": sum(z == 0 and key == "OP6" for z, key in zip(CELL_Z, CELL_KEYS)),
        "z0_B": sum(z == 0 and key == "OP7" for z, key in zip(CELL_Z, CELL_KEYS)),
        "z1_A": sum(z == 1 and key == "OP6" for z, key in zip(CELL_Z, CELL_KEYS)),
        "z1_B": sum(z == 1 and key == "OP7" for z, key in zip(CELL_Z, CELL_KEYS)),
    }
    if realized != {"z0_A": 8, "z0_B": 8, "z1_A": 8, "z1_B": 8}:
        raise RuntimeError(f"EXP2 live cells are not 8/8/8/8: {realized}")
    return {
        "exp2_protocol": {
            **contract,
            "resolved_live_cells": realized,
            "resolved_opponent_rows": rows,
            "ruleset_id": core.cfg.ruleset_id,
            "own_flag_home_required_to_score": True,
        }
    }


def _preflight_launch() -> None:
    if not IMPLEMENTATION_GATE.is_file():
        raise RuntimeError(f"implementation gate missing: {IMPLEMENTATION_GATE}")
    gate = json.loads(IMPLEMENTATION_GATE.read_text(encoding="utf-8"))
    if gate.get("verdict") != "PASS" or gate.get("production_launch_authorized") is not True:
        raise RuntimeError("implementation gate does not authorize production launch")
    run_dir = OUT / RUN_TAG
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(
            f"EXP2 one-attempt guard: run directory is not empty: {run_dir}. "
            "Do not resume, overwrite, or relaunch without first freezing a "
            "typed invalid-run record for a fail-fast engineering fault."
        )
    for path, expected in zip(TEACHER_PATHS, TEACHER_HASHES):
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"frozen teacher missing or hash mismatch: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true", help="spend the frozen training block")
    args = parser.parse_args()
    cfg, contract = build_exp2_config()
    print(json.dumps({
        "mode": "LAUNCH" if args.launch else "CONTRACT_ONLY",
        "run_tag": cfg.run_tag,
        "seed": cfg.seed,
        "steps": cfg.total_timesteps,
        "cells": contract["static_cells"],
        "teacher_sha256": contract["teacher_sha256"],
        "resolved_config_diff": contract["resolved_config_diff"],
    }, indent=2, default=str))
    if not args.launch:
        print("CONTRACT ONLY. No environment constructed and no training step spent.")
        return 0
    _preflight_launch()
    orchestrate_training_run(
        cfg,
        pre_rollout_env_setup=partial(configure_exp2_live_environment, contract=contract),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
