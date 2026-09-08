"""Synthetic, no-environment EXP2 lifecycle smoke with the frozen teachers.

Loads both immutable SAPPO checkpoints, constructs the production K=2 student
architecture and Adam optimizer, attaches the online KL runner after updater
construction, and proves one 1:4 cadence update lowers both mapped KLs. This
does not consume a training, development, or evaluation seed or environment
step and cannot produce a scientific result.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_exp2_k2_latent_compression import (  # noqa: E402
    SD,
    TEACHER_HASHES,
    TEACHER_PATHS,
    build_exp2_config,
)
from game_field_gpu import VEC_OBS_DIM  # noqa: E402
from rl.custom_ppo.exp2_teacher_compression import (  # noqa: E402
    Exp2TeacherCompressionRunner,
    teacher_student_kl,
)
from rl.custom_ppo.inference import load_custom_ppo_policy  # noqa: E402
from rl.custom_ppo.policy import SharedActorCentralizedCritic  # noqa: E402
from rl.custom_ppo.update.updater import PPOUpdater  # noqa: E402

OUT = SD / "EXP2_K2_TEACHER_KL_SMOKE.json"
SMOKE_SEED = 20260820


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spaces():
    obs = spaces.Dict({
        "grid": spaces.Box(0.0, 1.0, shape=(2, 7, 20, 20), dtype=np.float32),
        "vec": spaces.Box(-1.0, 1.0, shape=(2, VEC_OBS_DIM), dtype=np.float32),
        "agent_mask": spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32),
        "mask": spaces.Box(0.0, 1.0, shape=(110,), dtype=np.float32),
    })
    return obs, spaces.MultiDiscrete([5, 50, 5, 50])


def main() -> int:
    cfg, contract = build_exp2_config()
    for path, expected in zip(TEACHER_PATHS, TEACHER_HASHES):
        actual = _sha(path)
        if actual != expected:
            raise RuntimeError(f"frozen teacher hash mismatch: {path}: {actual}")

    torch.manual_seed(SMOKE_SEED)
    obs_space, action_space = _spaces()
    student = SharedActorCentralizedCritic(
        obs_space, action_space, latent_k=2,
        strategy_encoder_enabled=False, z_embed_dim=int(cfg.latent_z_embed_dim),
        strategy_hidden_dim=int(cfg.latent_strategy_hidden),
        critic_hidden_dim=int(cfg.latent_vf_hidden),
        actor_cnn_feature_dim=int(cfg.actor_cnn_feature_dim),
    )
    teachers = {
        z: load_custom_ppo_policy(
            str(path), obs_space, action_space, device="cpu"
        ).model
        for z, path in enumerate(TEACHER_PATHS)
    }
    optimizer = torch.optim.Adam(student.parameters(), lr=float(cfg.learning_rate))
    runner = Exp2TeacherCompressionRunner(
        student, optimizer, teachers,
        lambda_teacher=float(cfg.exp2_teacher_lambda),
        cadence=int(cfg.exp2_teacher_cadence),
        batch_size=int(cfg.exp2_teacher_batch_size),
        max_grad_norm=float(cfg.max_grad_norm),
        seed=SMOKE_SEED + 1,
        device="cpu",
    )

    batch_size = 64
    obs = {
        "grid": torch.rand(batch_size, 2, 7, 20, 20),
        "vec": torch.rand(batch_size, 2, VEC_OBS_DIM) * 2.0 - 1.0,
        "agent_mask": torch.ones(batch_size, 2),
        "mask": torch.ones(batch_size, 110),
    }
    z = torch.tensor([0] * 32 + [1] * 32, dtype=torch.long)
    decision = torch.ones(batch_size, 2, dtype=torch.bool)
    batch = {
        "obs_grid": obs["grid"], "obs_vec": obs["vec"],
        "obs_agent_mask": obs["agent_mask"], "obs_mask": obs["mask"],
        "z": z,
    }
    with torch.no_grad():
        _, before = teacher_student_kl(student, teachers, obs, z, decision)

    # Production lifecycle: the updater already exists when the runner is
    # attached to its runtime owner.
    runtime = SimpleNamespace(exp2_teacher_compression_runner=None)
    updater = PPOUpdater.__new__(PPOUpdater)
    updater.runtime = runtime
    if updater._exp2_teacher_runner() is not None:
        raise RuntimeError("EXP2 runner was present before attachment")
    runtime.exp2_teacher_compression_runner = runner
    for _ in range(4):
        updater._exp2_teacher_runner().note_ppo_minibatch(batch)
        updater._assert_exp2_teacher_cadence(runner)
    with torch.no_grad():
        _, after = teacher_student_kl(student, teachers, obs, z, decision)

    passed = (
        runner.n_ppo_actor_minibatches == 4
        and runner.n_teacher_updates == 1
        and after["kl_z0"] < before["kl_z0"]
        and after["kl_z1"] < before["kl_z1"]
        and student.strategy_encoder is None
    )
    artifact = {
        "artifact_id": "EXP2_K2_TEACHER_KL_SMOKE",
        "classification": "IMPLEMENTATION_SMOKE_NOT_A_SCIENTIFIC_RESULT",
        "environment_steps": 0,
        "seed_block_consumed": False,
        "smoke_seed": SMOKE_SEED,
        "protocol_id": contract["protocol_id"],
        "teacher_sha256": list(TEACHER_HASHES),
        "student_q_phi": "ABSENT",
        "teacher_mapping": contract["teacher_mapping"],
        "lambda": runner.lambda_teacher,
        "cadence": runner.cadence,
        "batch_size": runner.batch_size,
        "n_ppo_actor_updates": runner.n_ppo_actor_minibatches,
        "n_teacher_updates": runner.n_teacher_updates,
        "before": before,
        "after": after,
        "verdict": "PASS" if passed else "FAIL",
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    OUT.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2))
    if not passed:
        raise RuntimeError("EXP2 teacher-KL smoke failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
