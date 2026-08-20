"""Launch-blocking tests for frozen EXP2 K=2 supervised compression."""
from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces

from experiments.run_exp2_k2_latent_compression import (
    CELL_KEYS,
    CELL_Z,
    TEACHER_HASHES,
    build_exp2_config,
)
from game_field_gpu import VEC_OBS_DIM
from rl.custom_ppo.exp2_teacher_compression import (
    Exp2TeacherCompressionRunner,
    decision_eligible_agents,
    teacher_student_kl,
)
from rl.custom_ppo.latent.state import LatentStrategyState
from rl.custom_ppo.policy import SharedActorCentralizedCritic
from rl.custom_ppo.update.updater import PPOUpdater


def _obs_space():
    return spaces.Dict({
        "grid": spaces.Box(0.0, 1.0, shape=(2, 7, 20, 20), dtype=np.float32),
        "vec": spaces.Box(-1.0, 1.0, shape=(2, VEC_OBS_DIM), dtype=np.float32),
        "agent_mask": spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32),
        "mask": spaces.Box(0.0, 1.0, shape=(110,), dtype=np.float32),
    })


def _action_space():
    return spaces.MultiDiscrete([5, 50, 5, 50])


def _student():
    return SharedActorCentralizedCritic(
        _obs_space(), _action_space(), latent_k=2,
        strategy_encoder_enabled=False, z_embed_dim=16,
    )


def _teacher(seed: int):
    torch.manual_seed(seed)
    return SharedActorCentralizedCritic(_obs_space(), _action_space(), latent_k=0)


def _obs(batch: int = 8):
    return {
        "grid": torch.rand(batch, 2, 7, 20, 20),
        "vec": torch.rand(batch, 2, VEC_OBS_DIM),
        "agent_mask": torch.ones(batch, 2),
        "mask": torch.ones(batch, 110),
    }


def _batch(batch: int = 8):
    obs = _obs(batch)
    return {
        "obs_grid": obs["grid"],
        "obs_vec": obs["vec"],
        "obs_agent_mask": obs["agent_mask"],
        "obs_mask": obs["mask"],
        "z": torch.tensor(([0, 1] * (batch // 2)), dtype=torch.long),
    }


def test_frozen_config_and_cells_resolve_exactly():
    cfg, contract = build_exp2_config()
    assert cfg.use_latent_strategy and cfg.latent_k == 2
    assert cfg.latent_strategy_encoder_enabled is False
    assert cfg.latent_assignment_mode == "static_env"
    assert tuple(cfg.forced_latent_env_ids) == CELL_Z
    assert len(CELL_KEYS) == len(CELL_Z) == 32
    assert contract["static_cells"] == {"z0_A": 8, "z0_B": 8, "z1_A": 8, "z1_B": 8}
    assert tuple(cfg.exp2_teacher_sha256) == TEACHER_HASHES
    assert cfg.exp2_teacher_lambda == 0.10
    assert cfg.exp2_teacher_cadence == 4
    assert cfg.exp2_teacher_batch_size == 64
    assert cfg.load_path is None


def test_shared_actor_receives_z_while_q_phi_is_structurally_absent():
    torch.manual_seed(10)
    model = _student()
    assert model.strategy_encoder is None
    assert model.uses_latent_strategy
    assert len(list(model.latent_actor.parameters())) > 0
    obs = _obs(2)
    # Make the two embedding rows observably distinct without changing any
    # shared actor parameter or introducing per-mode heads.
    with torch.no_grad():
        model.latent_actor.strategy_embedding.weight[0].fill_(-1.0)
        model.latent_actor.strategy_embedding.weight[1].fill_(1.0)
    logits0 = model.policy_logits(obs, z_idx=torch.zeros(2, dtype=torch.long))
    logits1 = model.policy_logits(obs, z_idx=torch.ones(2, dtype=torch.long))
    assert not torch.equal(logits0, logits1), "z did not reach the shared actor"


def test_decision_eligibility_uses_the_same_legal_mask_contract():
    mask = torch.ones(2, 110)
    # Lock agent 1's macro head in row 0 to one legal action.
    mask[0, 55:60] = 0.0
    mask[0, 55] = 1.0
    alive = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    got = decision_eligible_agents(
        mask, action_dims=(5, 50, 5, 50), n_agents=2, agent_mask=alive,
    )
    assert got.tolist() == [[True, False], [True, False]]


def test_teacher_kl_is_teacher_to_student_and_routes_only_by_z():
    torch.manual_seed(11)
    student = _student()
    teachers = {0: _teacher(20), 1: _teacher(21)}
    obs = _obs(8)
    z = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    decision = torch.ones(8, 2, dtype=torch.bool)
    loss, metrics = teacher_student_kl(student, teachers, obs, z, decision)
    assert loss.requires_grad
    assert metrics["rows_z0"] == metrics["rows_z1"] == 4.0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())
    assert all(p.grad is None for teacher in teachers.values() for p in teacher.parameters())


def test_integrated_lifecycle_smoke_decreases_both_mapped_kls():
    """Late attach, four PPO completions, one real optimizer KL update."""
    torch.manual_seed(12)
    student = _student()
    teachers = {0: _teacher(30), 1: _teacher(31)}
    optimizer = torch.optim.SGD(student.parameters(), lr=0.05)
    batch = _batch(8)
    obs = {
        "grid": batch["obs_grid"], "vec": batch["obs_vec"],
        "agent_mask": batch["obs_agent_mask"], "mask": batch["obs_mask"],
    }
    decision = torch.ones(8, 2, dtype=torch.bool)
    before, before_t = teacher_student_kl(student, teachers, obs, batch["z"], decision)
    teacher_before = {
        z: [p.detach().clone() for p in teacher.parameters()]
        for z, teacher in teachers.items()
    }

    runtime = SimpleNamespace(exp2_teacher_compression_runner=None)
    updater = PPOUpdater.__new__(PPOUpdater)
    updater.runtime = runtime
    assert updater._exp2_teacher_runner() is None
    runner = Exp2TeacherCompressionRunner(
        student, optimizer, teachers, lambda_teacher=0.10, cadence=4,
        batch_size=64, max_grad_norm=None, seed=99, device="cpu",
    )
    runtime.exp2_teacher_compression_runner = runner
    for _ in range(4):
        updater._exp2_teacher_runner().note_ppo_minibatch(batch)
        updater._assert_exp2_teacher_cadence(runner)

    after, after_t = teacher_student_kl(student, teachers, obs, batch["z"], decision)
    assert runner.n_ppo_actor_minibatches == 4
    assert runner.n_teacher_updates == 1
    assert runner.telemetry()["exp2_teacher_to_ppo_ratio"] == 0.25
    assert float(after.detach()) < float(before.detach())
    assert after_t["kl_z0"] < before_t["kl_z0"]
    assert after_t["kl_z1"] < before_t["kl_z1"]
    for z, teacher in teachers.items():
        assert all(torch.equal(old, new) for old, new in zip(teacher_before[z], teacher.parameters()))


def test_runner_resume_preserves_cadence_and_rng_state():
    student = _student()
    teachers = {0: _teacher(40), 1: _teacher(41)}
    runner = Exp2TeacherCompressionRunner(
        student, torch.optim.SGD(student.parameters(), lr=0.01), teachers,
        lambda_teacher=0.10, cadence=4, batch_size=64,
        max_grad_norm=None, seed=1, device="cpu",
    )
    for _ in range(7):
        runner.note_ppo_minibatch(_batch(8))
    state = copy.deepcopy(runner.state_dict())
    restored_student = _student()
    restored = Exp2TeacherCompressionRunner(
        restored_student, torch.optim.SGD(restored_student.parameters(), lr=0.01),
        {0: _teacher(40), 1: _teacher(41)}, lambda_teacher=0.10,
        cadence=4, batch_size=64, max_grad_norm=None, seed=2, device="cpu",
    )
    restored.load_state_dict(state)
    assert restored.n_ppo_actor_minibatches == 7
    assert restored.n_teacher_updates == 1
    restored.note_ppo_minibatch(_batch(8))
    assert restored.n_teacher_updates == 2


def test_disabled_treatment_has_no_runner_or_teacher_path():
    from rl.training.orchestrator import _maybe_attach_exp2_teacher_compression

    trainer = SimpleNamespace(marker="untouched")
    _maybe_attach_exp2_teacher_compression(
        SimpleNamespace(exp2_teacher_compression_enabled=False), trainer
    )
    assert not hasattr(trainer, "exp2_teacher_compression_runner")
    with pytest.raises(ValueError, match="lambda must be > 0"):
        Exp2TeacherCompressionRunner(
            _student(), None, {0: _teacher(1), 1: _teacher(2)},
            lambda_teacher=0.0, cadence=4, batch_size=64,
            max_grad_norm=None, seed=1, device="cpu",
        )


def test_static_env_assignment_persists_through_episode_reset():
    from tests.test_latent_episode_warmup import _make_trainer

    trainer = _make_trainer(n_envs=4, warmup=0, episode_credit=False, gs_dim=4)
    trainer.latent_k = 2
    trainer.cfg.latent_assignment_mode = "static_env"
    trainer.cfg.forced_latent_env_ids = (0, 0, 1, 1)
    state = LatentStrategyState(trainer)
    state.reset()
    gs = torch.zeros(4, 4)
    z0, _, _ = state.strategy_for_step(gs)
    z1, _, _ = state.strategy_for_step(gs)
    assert z0.tolist() == z1.tolist() == [0, 0, 1, 1]
    state.mark_strategy_step_done(np.asarray([True, False, True, False]))
    z2, _, aux = state.strategy_for_step(gs)
    assert z2.tolist() == [0, 0, 1, 1]
    assert aux["z_resampled"].tolist() == [True, False, True, False]


def test_cadence_invariant_aborts_on_silent_noop():
    silent = SimpleNamespace(cadence=4, n_ppo_actor_minibatches=4, n_teacher_updates=0)
    with pytest.raises(RuntimeError, match="EXP2 teacher cadence violated"):
        PPOUpdater._assert_exp2_teacher_cadence(silent)


def test_telemetry_columns_are_unconditional_and_last_aggregated():
    from rl.custom_ppo.csv_writers import _update_fieldnames
    from rl.custom_ppo.update.telemetry import AggregationMode, DEFAULT_METRIC_SCHEMA

    required = {
        "exp2_n_ppo_actor_updates", "exp2_n_teacher_updates",
        "exp2_teacher_to_ppo_ratio", "exp2_teacher_loss",
        "exp2_teacher_kl", "exp2_teacher_kl_z0", "exp2_teacher_kl_z1",
        "exp2_teacher_agreement_z0", "exp2_teacher_agreement_z1",
        "exp2_cell_steps_z0_A", "exp2_cell_steps_z0_B",
        "exp2_cell_steps_z1_A", "exp2_cell_steps_z1_B",
    }
    assert required <= set(_update_fieldnames(False, 0))
    assert all(DEFAULT_METRIC_SCHEMA[name] is AggregationMode.LAST for name in required)


def test_student_checkpoint_roundtrip_preserves_router_absence():
    from rl.custom_ppo import (
        CUSTOM_PPO_ACTOR_ARCH,
        CUSTOM_PPO_LATENT_FORMAT,
        CUSTOM_PPO_VEC_SCHEMA_VERSION,
    )
    from rl.custom_ppo.inference import load_custom_ppo_checkpoint

    model = _student()
    cfg = {
        "seed": 1, "max_blue_agents": 2, "use_latent_strategy": True,
        "latent_k": 2, "latent_strategy_encoder_enabled": False,
        "latent_z_embed_dim": 16, "latent_strategy_hidden": 128,
        "latent_vf_hidden": 128, "latent_strategy_aux_return_head": False,
        "latent_episode_strategy_ppo": False, "actor_cnn_feature_dim": 128,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "exp2_student.zip"
        torch.save({
            "model_state_dict": model.state_dict(), "cfg": cfg,
            "format": CUSTOM_PPO_LATENT_FORMAT,
            "actor_arch": CUSTOM_PPO_ACTOR_ARCH,
            "actor_cnn_feature_dim": 128,
            "global_state_dim": int(model.global_state_dim),
            "vec_schema_version": CUSTOM_PPO_VEC_SCHEMA_VERSION,
        }, path)
        loaded = load_custom_ppo_checkpoint(
            str(path), _obs_space(), _action_space(), device="cpu"
        ).policy.model
    assert loaded.uses_latent_strategy and loaded.latent_k == 2
    assert loaded.strategy_encoder is None
    assert loaded.strategy_encoder_enabled is False
