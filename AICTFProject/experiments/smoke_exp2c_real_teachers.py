"""Zero-environment EXP2C smoke with both frozen SAPPO teachers."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_exp2_k2_latent_compression import TEACHER_HASHES, TEACHER_PATHS
from experiments.run_exp2c_mode_specific_actor_compression import SD, build_exp2c_config
from experiments.smoke_exp2_teacher_compression import _spaces
from game_field_gpu import VEC_OBS_DIM
from rl.custom_ppo.exp2_teacher_compression import (
    Exp2TeacherCompressionRunner,
    _kl_for_rows,
    teacher_student_kl,
)
from rl.custom_ppo.inference import load_custom_ppo_policy
from rl.custom_ppo.policy import SharedActorCentralizedCritic
from rl.custom_ppo.update.updater import PPOUpdater

OUT = SD / "EXP2C_REAL_TEACHER_SMOKE.json"
SMOKE_SEED = 20260823


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _norm(grads) -> float:
    tensors = [g.reshape(-1) for g in grads if g is not None]
    return float(torch.cat(tensors).norm()) if tensors else 0.0


def main() -> int:
    cfg, contract = build_exp2c_config()
    for path, expected in zip(TEACHER_PATHS, TEACHER_HASHES):
        if _sha(path) != expected:
            raise RuntimeError(f"frozen teacher hash mismatch: {path}")
    torch.manual_seed(SMOKE_SEED)
    obs_space, action_space = _spaces()
    student = SharedActorCentralizedCritic(
        obs_space, action_space, latent_k=2,
        strategy_encoder_enabled=False, z_embed_dim=int(cfg.latent_z_embed_dim),
        strategy_hidden_dim=int(cfg.latent_strategy_hidden),
        critic_hidden_dim=int(cfg.latent_vf_hidden),
        actor_cnn_feature_dim=int(cfg.actor_cnn_feature_dim),
        exp2c_mode_specific_action_heads=True,
    )
    teachers = {
        z: load_custom_ppo_policy(str(path), obs_space, action_space, device="cpu").model
        for z, path in enumerate(TEACHER_PATHS)
    }
    obs = {
        "grid": torch.rand(64, 2, 7, 20, 20),
        "vec": torch.rand(64, 2, VEC_OBS_DIM) * 2.0 - 1.0,
        "agent_mask": torch.ones(64, 2),
        "mask": torch.ones(64, 110),
    }
    z = torch.tensor([0] * 32 + [1] * 32, dtype=torch.long)
    decision = torch.ones(64, 2, dtype=torch.bool)
    heads = student.latent_actor.latent_action_heads
    routing = {}
    for mode in (0, 1):
        rows = torch.arange(mode * 32, (mode + 1) * 32)
        obs_z = {key: value.index_select(0, rows) for key, value in obs.items()}
        loss = _kl_for_rows(
            teachers[mode], student, obs_z,
            z_idx=torch.full((32,), mode, dtype=torch.long),
            decision_mask=decision.index_select(0, rows),
        ).kl
        own = torch.autograd.grad(loss, list(heads[mode].parameters()), retain_graph=True)
        other = torch.autograd.grad(
            loss, list(heads[1 - mode].parameters()), allow_unused=True,
        )
        routing[f"z{mode}_own_head_grad_norm"] = _norm(own)
        routing[f"z{mode}_other_head_grad_norm"] = _norm(other)

    optimizer = torch.optim.Adam(student.parameters(), lr=float(cfg.learning_rate))
    runner = Exp2TeacherCompressionRunner(
        student, optimizer, teachers,
        lambda_teacher=float(cfg.exp2_teacher_lambda),
        cadence=int(cfg.exp2_teacher_cadence),
        batch_size=int(cfg.exp2_teacher_batch_size),
        max_grad_norm=float(cfg.max_grad_norm), seed=SMOKE_SEED + 1, device="cpu",
        cell_counts=(16, 0, 0, 16), gradient_cosine_enabled=False,
        clip_range=float(cfg.clip_range),
    )
    batch = {
        "obs_grid": obs["grid"], "obs_vec": obs["vec"],
        "obs_agent_mask": obs["agent_mask"], "obs_mask": obs["mask"], "z": z,
    }
    with torch.no_grad():
        _, before = teacher_student_kl(student, teachers, obs, z, decision)
    head_before = [
        {name: value.detach().clone() for name, value in head.state_dict().items()}
        for head in heads
    ]
    runtime = SimpleNamespace(exp2_teacher_compression_runner=None)
    updater = PPOUpdater.__new__(PPOUpdater)
    updater.runtime = runtime
    runtime.exp2_teacher_compression_runner = runner
    for _ in range(4):
        updater._exp2_teacher_runner().note_ppo_minibatch(batch)
        updater._assert_exp2_teacher_cadence(runner)
    with torch.no_grad():
        _, after = teacher_student_kl(student, teachers, obs, z, decision)
    head_deltas = [
        max(float((value - head_before[mode][name]).abs().max()) for name, value in head.state_dict().items())
        for mode, head in enumerate(heads)
    ]
    passed = (
        runner.n_ppo_actor_minibatches == 4
        and runner.n_teacher_updates == 1
        and after["kl_z0"] < before["kl_z0"]
        and after["kl_z1"] < before["kl_z1"]
        and all(value > 0.0 for value in head_deltas)
        and routing["z0_own_head_grad_norm"] > 0.0
        and routing["z1_own_head_grad_norm"] > 0.0
        and routing["z0_other_head_grad_norm"] == 0.0
        and routing["z1_other_head_grad_norm"] == 0.0
        and student.latent_actor.latent_adapters is None
        and student.latent_actor.latent_branch_trunks is None
    )
    artifact = {
        "artifact_id": "EXP2C_REAL_TEACHER_SMOKE",
        "classification": "IMPLEMENTATION_SMOKE_NOT_A_SCIENTIFIC_RESULT",
        "environment_steps": 0,
        "seed_block_consumed": False,
        "smoke_seed": SMOKE_SEED,
        "protocol_id": contract["protocol_id"],
        "teacher_sha256": list(TEACHER_HASHES),
        "architecture": "shared CNN/body + exactly two private final linear heads",
        "private_adapters": "ABSENT",
        "private_deep_trunks": "ABSENT",
        "routing": routing,
        "head_max_abs_deltas": head_deltas,
        "before": before,
        "after": after,
        "n_ppo_actor_updates": runner.n_ppo_actor_minibatches,
        "n_teacher_updates": runner.n_teacher_updates,
        "verdict": "PASS" if passed else "FAIL",
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    OUT.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2))
    if not passed:
        raise RuntimeError("EXP2C real-teacher smoke failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
