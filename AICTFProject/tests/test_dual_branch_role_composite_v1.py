"""IMPLEMENTATION_GATE tests for DUAL_BRANCH_ROLE_COMPOSITE_V1.

Both ATTACK and DEFEND branches are trainable, initialized from the same
foundation specialist. ATTACK samples must not update DEFEND weights and
vice versa. Teacher loss must never enter the ATTACK branch.
k = ceil(N/3) is enforced.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from pathlib import Path

import pytest
import torch

from rl.custom_ppo.split_attack_defend import (
    attack_gated_sum,
    ceil_n_over_3,
    defend_gated_sum,
)


def test_ceil_n_over_3_locked_table():
    assert ceil_n_over_3(2) == 1
    assert ceil_n_over_3(4) == 2
    assert ceil_n_over_3(6) == 2


def test_attack_gated_sum_complements_defend_gated_sum():
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    is_defend = torch.tensor([[True, False, True, False]])
    d = defend_gated_sum(values, is_defend)
    a = attack_gated_sum(values, is_defend)
    assert torch.allclose(d, torch.tensor([1.0 + 3.0]))
    assert torch.allclose(a, torch.tensor([2.0 + 4.0]))
    assert torch.allclose(d + a, values.sum(dim=-1))


def _build_env(seed: int):
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig

    return GPUCTFVecEnv(
        GPUFieldConfig(
            n_envs=1, n_agents_per_team=2, max_decision_steps=100, device="cpu", seed=seed
        )
    )


def _base_cfg(*, run_tag: str, checkpoint_dir: str):
    from rl.config.ppo_config import PPOConfig

    cfg = PPOConfig()
    cfg.seed = 0
    cfg.total_timesteps = 16
    cfg.n_envs = 1
    cfg.n_steps = 16
    cfg.batch_size = 16
    cfg.n_epochs = 1
    cfg.use_stable_marl_ppo = False
    cfg.device = "cpu"
    cfg.enable_tensorboard = False
    cfg.enable_checkpoints = False
    cfg.enable_eval = False
    cfg.verbose_training = False
    cfg.max_blue_agents = 2
    cfg.mode = "FIXED_OPPONENT"
    cfg.fixed_opponent_tag = "OP3"
    cfg.use_latent_strategy = False
    cfg.gpu_native_env = True
    cfg.run_tag = run_tag
    cfg.checkpoint_dir = checkpoint_dir
    return cfg


def _save_foundation(tmp_path: Path, *, seed: int, name: str) -> Path:
    from rl.custom_ppo.checkpoints.loader import save_trainer_checkpoint
    from rl.custom_ppo.trainer import CustomPPOTrainer

    env = _build_env(seed=seed)
    try:
        cfg = _base_cfg(run_tag=f"dual_src_{name}", checkpoint_dir=str(tmp_path))
        trainer = CustomPPOTrainer(
            env, cfg, learning_rate=1e-4, clip_range=0.2, ent_coef=0.0,
            n_epochs=1, batch_size=16, value_clip_range=0.2,
        )
        path = tmp_path / f"foundation_{name}.zip"
        save_trainer_checkpoint(trainer, str(path))
    finally:
        env.close()
    return path


def _build_dual_trainer(tmp_path: Path, foundation: Path, *, seed: int, name: str):
    from rl.custom_ppo.trainer import CustomPPOTrainer
    from rl.training.orchestrator import (
        _maybe_attach_defend_teacher,
        _maybe_attach_split_attack_defend,
    )

    env = _build_env(seed=seed)
    cfg = _base_cfg(run_tag=f"dual_tgt_{name}", checkpoint_dir=str(tmp_path))
    cfg.role_conditioning_enabled = True
    cfg.role_hold_ticks = 8
    cfg.role_fixed_for_episode = True
    cfg.role_k_defend = 1  # ceil(2/3)
    cfg.split_attack_defend_enabled = True
    cfg.split_attack_defend_frozen_ckpt = str(foundation)
    cfg.dual_branch_role_composite_enabled = True
    cfg.defend_teacher_lambda = 0.1
    cfg.defend_teacher_lambda_end = 0.0
    cfg.defend_teacher_decay_start_step = 50_000
    cfg.defend_teacher_decay_end_step = 150_000
    cfg.defend_teacher_cadence = 1

    trainer = CustomPPOTrainer(
        env, cfg, learning_rate=1e-3, clip_range=0.2, ent_coef=0.01,
        n_epochs=2, batch_size=16, value_clip_range=0.2,
    )
    _maybe_attach_split_attack_defend(cfg, trainer)
    _maybe_attach_defend_teacher(cfg, trainer)
    return trainer, env, cfg


def test_dual_branch_both_trainable_and_isolated(tmp_path):
    foundation = _save_foundation(tmp_path, seed=11, name="iso")
    trainer, env, cfg = _build_dual_trainer(tmp_path, foundation, seed=12, name="iso")
    try:
        attack = trainer.dual_branch_attack_model
        defend = trainer.model
        assert attack is not None
        assert trainer.dual_branch_attack_optimizer is not None
        assert all(p.requires_grad for p in attack.parameters())
        assert trainer.defend_teacher_runner is not None
        assert trainer.defend_teacher_runner.student is defend

        attack_before = [p.detach().clone() for p in attack.parameters()]
        defend_before = [p.detach().clone() for p in defend.parameters()]

        rollout = trainer.collect_rollout()
        assert "attack_log_probs" in rollout.fields
        assert "defend_log_probs" in rollout.fields
        assert "attack_values_norm" in rollout.fields

        stats = trainer.update(rollout, total_timesteps=int(cfg.total_timesteps))
        assert stats is not None

        attack_changed = any(
            not torch.equal(a, b) for a, b in zip(attack_before, attack.parameters())
        )
        defend_changed = any(
            not torch.equal(a, b) for a, b in zip(defend_before, defend.parameters())
        )
        assert attack_changed, "ATTACK branch must receive PPO updates"
        assert defend_changed, "DEFEND branch must receive PPO updates"

        teacher_param_ids = {id(p) for p in attack.parameters()}
        for group in trainer.defend_teacher_runner.optimizer.param_groups:
            for p in group["params"]:
                assert id(p) not in teacher_param_ids, (
                    "DEFEND teacher optimizer must not own ATTACK parameters"
                )
    finally:
        env.close()


def test_dual_branch_refuses_wrong_k(tmp_path):
    foundation = _save_foundation(tmp_path, seed=21, name="k")
    from rl.custom_ppo.trainer import CustomPPOTrainer
    from rl.training.orchestrator import _maybe_attach_split_attack_defend

    env = _build_env(seed=22)
    try:
        cfg = _base_cfg(run_tag="dual_bad_k", checkpoint_dir=str(tmp_path))
        cfg.role_conditioning_enabled = True
        cfg.role_fixed_for_episode = True
        cfg.role_k_defend = 2  # wrong for N=2
        cfg.split_attack_defend_enabled = True
        cfg.split_attack_defend_frozen_ckpt = str(foundation)
        cfg.dual_branch_role_composite_enabled = True
        trainer = CustomPPOTrainer(
            env, cfg, learning_rate=1e-3, clip_range=0.2, ent_coef=0.0,
            n_epochs=1, batch_size=16, value_clip_range=0.2,
        )
        with pytest.raises(RuntimeError, match=r"ceil\(N/3\)"):
            _maybe_attach_split_attack_defend(cfg, trainer)
    finally:
        env.close()


def test_dual_branch_checkpoint_roundtrip_persists_attack(tmp_path):
    from rl.custom_ppo.checkpoints.loader import load_trainer_checkpoint, save_trainer_checkpoint
    from rl.custom_ppo.trainer import CustomPPOTrainer
    from rl.training.orchestrator import _maybe_attach_split_attack_defend

    foundation = _save_foundation(tmp_path, seed=31, name="ck")
    trainer, env, cfg = _build_dual_trainer(tmp_path, foundation, seed=32, name="ck")
    try:
        rollout = trainer.collect_rollout()
        trainer.update(rollout, total_timesteps=int(cfg.total_timesteps))
        attack_hash = tuple(
            p.detach().cpu().sum().item() for p in trainer.dual_branch_attack_model.parameters()
        )
        out = tmp_path / "dual_ckpt.zip"
        save_trainer_checkpoint(trainer, str(out))
    finally:
        env.close()

    env2 = _build_env(seed=33)
    try:
        cfg2 = _base_cfg(run_tag="dual_resume", checkpoint_dir=str(tmp_path))
        cfg2.role_conditioning_enabled = True
        cfg2.role_hold_ticks = 8
        cfg2.role_fixed_for_episode = True
        cfg2.role_k_defend = 1
        cfg2.split_attack_defend_enabled = True
        cfg2.split_attack_defend_frozen_ckpt = str(foundation)
        cfg2.dual_branch_role_composite_enabled = True
        t2 = CustomPPOTrainer(
            env2, cfg2, learning_rate=1e-3, clip_range=0.2, ent_coef=0.01,
            n_epochs=1, batch_size=16, value_clip_range=0.2,
        )
        load_trainer_checkpoint(t2, str(out), reset_progress=False)
        _maybe_attach_split_attack_defend(cfg2, t2)
        assert t2.dual_branch_attack_model is not None
        loaded = tuple(
            p.detach().cpu().sum().item() for p in t2.dual_branch_attack_model.parameters()
        )
        assert loaded == attack_hash
    finally:
        env2.close()
