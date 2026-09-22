"""Executable contracts for DEFEND_ATTACK_SPLIT_POLICY_A_V1_SPEC.

Two physically separate networks instead of one shared role-conditioned
model: a frozen pi_A produces ATTACK-role actions (no optimizer, no
gradient path, ever), a trainable model produces DEFEND-role actions, with
the main PPO actor loss and entropy bonus gated to DEFEND-role agent slots
only. See DEFEND_TEACHER_ROLE_CONDITIONING_A_V1's own negative result
(catastrophic interference from a single shared network) for the
motivation.

Guard/isolation contracts are unit-level (SimpleNamespace mocks). The
splice mechanism and the DEFEND-gated main PPO loss are exercised
end-to-end against a real (tiny, cpu) env + trainer, because the PI's own
stated risk ("do not fake this by running one policy and overwriting
defender actions afterward") is precisely a wiring-order bug a pure unit
test of the gating math alone would not catch.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rl.custom_ppo.split_attack_defend import (
    defend_gated_sum,
    role_broadcast_mask,
    splice_actions,
)


# ---------------------------------------------------------------------------
# C-a/C-c/C-d: pure splice + DEFEND-gating math, exact and RNG-free
# ---------------------------------------------------------------------------


def test_splice_actions_defend_slots_from_trainable_attack_slots_from_frozen():
    """C-a: executed ATTACK actions come from frozen pi_A, executed DEFEND
    actions come from pi_D -- exact, with distinguishable sentinel values so
    a mix-up in either direction is unmistakable."""
    heads_per_agent = 2
    n_agents = 4
    batch = 3
    trainable = torch.full((batch, n_agents * heads_per_agent), 11, dtype=torch.long)
    frozen = torch.full((batch, n_agents * heads_per_agent), 99, dtype=torch.long)
    # agents 0,2 DEFEND; 1,3 ATTACK.
    is_defend = torch.tensor([[True, False, True, False]] * batch)

    exec_actions = splice_actions(trainable, frozen, is_defend, heads_per_agent)
    view = exec_actions.view(batch, n_agents, heads_per_agent)
    assert torch.equal(view[:, 0, :], torch.full((batch, heads_per_agent), 11))
    assert torch.equal(view[:, 2, :], torch.full((batch, heads_per_agent), 11))
    assert torch.equal(view[:, 1, :], torch.full((batch, heads_per_agent), 99))
    assert torch.equal(view[:, 3, :], torch.full((batch, heads_per_agent), 99))


def test_splice_actions_rejects_shape_mismatch():
    heads_per_agent = 2
    is_defend = torch.tensor([[True, False]])
    trainable = torch.zeros(1, 4, dtype=torch.long)
    frozen_wrong = torch.zeros(1, 3, dtype=torch.long)
    with pytest.raises(ValueError):
        splice_actions(trainable, frozen_wrong, is_defend, heads_per_agent)


def test_role_broadcast_mask_repeats_role_bit_across_heads():
    is_defend = torch.tensor([[True, False, True]])
    mask = role_broadcast_mask(is_defend, heads_per_agent=2)
    assert mask.tolist() == [[True, True, False, False, True, True]]


def test_defend_gated_sum_matches_manual_computation():
    """C-c: defend_log_probs equal the sum of only the DEFEND agents'
    per-agent log-probs (macro+waypoint already summed per agent by
    _log_prob_entropy_per_agent upstream of this function)."""
    per_agent = torch.tensor([[-1.0, -2.0, -3.0, -4.0], [-0.5, -0.5, -0.5, -0.5]])
    is_defend = torch.tensor([[True, False, True, False], [False, True, False, True]])
    got = defend_gated_sum(per_agent, is_defend)
    expected = torch.tensor([-1.0 + -3.0, -0.5 + -0.5])
    assert torch.allclose(got, expected)


def test_defend_gated_sum_ignores_attack_slot_values_entirely():
    """C-d: ATTACK-slot log-probs/entropy do not affect the gated scalar at
    all -- changing them arbitrarily must not change the result."""
    is_defend = torch.tensor([[True, False, True, False]])
    base = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
    perturbed = base.clone()
    perturbed[0, 1] = 12345.0  # ATTACK slot (index 1)
    perturbed[0, 3] = -99999.0  # ATTACK slot (index 3)
    assert torch.equal(defend_gated_sum(base, is_defend), defend_gated_sum(perturbed, is_defend))


# ---------------------------------------------------------------------------
# C: mutual exclusion / isolation (unit-level, no real model needed)
# ---------------------------------------------------------------------------


def test_isolation_split_attack_defend_absent_by_default():
    from rl.config.ppo_config import PPOConfig

    cfg = PPOConfig()
    assert bool(cfg.split_attack_defend_enabled) is False
    assert cfg.split_attack_defend_frozen_ckpt == ""


def test_maybe_attach_split_attack_defend_noop_when_disabled():
    from rl.training.orchestrator import _maybe_attach_split_attack_defend

    cfg = SimpleNamespace(split_attack_defend_enabled=False)
    trainer = SimpleNamespace()
    _maybe_attach_split_attack_defend(cfg, trainer)
    assert not hasattr(trainer, "split_attack_defend_frozen_model")


def test_maybe_attach_split_attack_defend_requires_role_conditioning():
    from rl.training.orchestrator import _maybe_attach_split_attack_defend

    cfg = SimpleNamespace(split_attack_defend_enabled=True, role_conditioning_enabled=False)
    with pytest.raises(RuntimeError, match="role_conditioning_enabled"):
        _maybe_attach_split_attack_defend(cfg, SimpleNamespace())


def test_maybe_attach_split_attack_defend_requires_fixed_for_episode():
    from rl.training.orchestrator import _maybe_attach_split_attack_defend

    cfg = SimpleNamespace(
        split_attack_defend_enabled=True, role_conditioning_enabled=True,
        role_fixed_for_episode=False,
    )
    with pytest.raises(RuntimeError, match="role_fixed_for_episode"):
        _maybe_attach_split_attack_defend(cfg, SimpleNamespace())


def test_maybe_attach_split_attack_defend_requires_ckpt():
    from rl.training.orchestrator import _maybe_attach_split_attack_defend

    cfg = SimpleNamespace(
        split_attack_defend_enabled=True, role_conditioning_enabled=True,
        role_fixed_for_episode=True, split_attack_defend_frozen_ckpt="",
    )
    with pytest.raises(RuntimeError, match="split_attack_defend_frozen_ckpt"):
        _maybe_attach_split_attack_defend(cfg, SimpleNamespace())


@pytest.mark.parametrize(
    "lam_field", ["sibling_sep_lambda", "role_pres_lambda", "getflag_preserve_lambda"],
)
def test_maybe_attach_split_attack_defend_rejects_other_aux_losses(lam_field):
    from rl.training.orchestrator import _maybe_attach_split_attack_defend

    cfg = SimpleNamespace(
        split_attack_defend_enabled=True, role_conditioning_enabled=True,
        role_fixed_for_episode=True, split_attack_defend_frozen_ckpt="x",
        **{lam_field: 0.1},
        **{f: 0.0 for f in ("sibling_sep_lambda", "role_pres_lambda", "getflag_preserve_lambda") if f != lam_field},
    )
    with pytest.raises(RuntimeError):
        _maybe_attach_split_attack_defend(cfg, SimpleNamespace())


@pytest.mark.parametrize(
    "attr", ["sappo_anchor_runner", "exp2_teacher_compression_runner",
             "sibling_sep_runner", "role_pres_runner", "getflag_preserve_runner"],
)
def test_maybe_attach_split_attack_defend_mutual_exclusion(attr):
    from rl.training.orchestrator import _maybe_attach_split_attack_defend

    cfg = SimpleNamespace(
        split_attack_defend_enabled=True, role_conditioning_enabled=True,
        role_fixed_for_episode=True, split_attack_defend_frozen_ckpt="x",
        sibling_sep_lambda=0.0, role_pres_lambda=0.0, getflag_preserve_lambda=0.0,
    )
    trainer = SimpleNamespace(**{attr: object()})
    with pytest.raises(RuntimeError):
        _maybe_attach_split_attack_defend(cfg, trainer)


@pytest.mark.parametrize(
    "fn_name,extra_kwargs",
    [
        ("_maybe_attach_sibling_separation", dict(
            sibling_sep_lambda=0.1, sibling_sep_ckpt="x", sibling_sep_dataset="y",
        )),
        ("_maybe_attach_role_preservation", dict(
            role_pres_lambda=0.1, role_pres_targets="x", role_pres_style="GUARD",
            sibling_sep_lambda=0.0, getflag_preserve_lambda=0.0,
        )),
        ("_maybe_attach_getflag_preservation", dict(
            getflag_preserve_lambda=0.1, getflag_preserve_ckpt="x",
            sibling_sep_lambda=0.0, role_pres_lambda=0.0,
        )),
    ],
)
def test_other_aux_losses_reject_split_attack_defend(fn_name, extra_kwargs):
    import rl.training.orchestrator as orch

    fn = getattr(orch, fn_name)
    cfg = SimpleNamespace(defend_teacher_lambda=0.0, split_attack_defend_enabled=True, **extra_kwargs)
    with pytest.raises(RuntimeError, match="split_attack_defend"):
        fn(cfg, SimpleNamespace())


# ---------------------------------------------------------------------------
# C: end-to-end integration -- real env, real frozen checkpoint, real update
# ---------------------------------------------------------------------------


def _build_env(seed: int):
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig

    return GPUCTFVecEnv(GPUFieldConfig(n_envs=1, n_agents_per_team=2, max_decision_steps=100,
                                        device="cpu", seed=seed))


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
    cfg.enable_progress_bar = False
    return cfg


def _state_dict_hash(model) -> str:
    import hashlib

    h = hashlib.sha256()
    sd = model.state_dict()
    for key in sorted(sd):
        h.update(key.encode("utf-8"))
        h.update(sd[key].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _save_frozen_checkpoint(tmp_path: Path, *, seed: int, name: str) -> tuple[Path, list]:
    """Build a plain (non-role-conditioned) trainer, save it as a checkpoint,
    and return (path, params-at-save) for later bit-identity comparison."""
    from rl.custom_ppo.checkpoints.loader import save_trainer_checkpoint
    from rl.custom_ppo.trainer import CustomPPOTrainer

    env_source = _build_env(seed=seed)
    try:
        cfg_source = _base_cfg(run_tag=f"split_ad_source_{name}", checkpoint_dir=str(tmp_path))
        source_trainer = CustomPPOTrainer(
            env_source, cfg_source, learning_rate=1e-4, clip_range=0.2, ent_coef=0.0,
            n_epochs=1, batch_size=16, value_clip_range=0.2,
        )
        ckpt_path = tmp_path / f"pi_a_frozen_{name}.zip"
        save_trainer_checkpoint(source_trainer, str(ckpt_path))
        params_at_save = [p.detach().clone() for p in source_trainer.model.parameters()]
    finally:
        env_source.close()
    return ckpt_path, params_at_save


def _build_split_trainer(tmp_path: Path, ckpt_path: Path, *, seed: int, name: str, attach_teacher: bool = False):
    """Construct a split-attack-defend trainer pointed at ckpt_path, attach
    the frozen model, and (optionally) also attach the DEFEND-teacher runner
    to exercise the combined path (C-i)."""
    from rl.custom_ppo.trainer import CustomPPOTrainer
    from rl.training.orchestrator import _maybe_attach_split_attack_defend

    env = _build_env(seed=seed)
    cfg = _base_cfg(run_tag=f"split_ad_target_{name}", checkpoint_dir=str(tmp_path))
    cfg.role_conditioning_enabled = True
    cfg.role_hold_ticks = 8
    cfg.role_fixed_for_episode = True
    cfg.split_attack_defend_enabled = True
    cfg.split_attack_defend_frozen_ckpt = str(ckpt_path)
    if attach_teacher:
        cfg.defend_teacher_lambda = 0.1

    trainer = CustomPPOTrainer(
        env, cfg, learning_rate=1e-3, clip_range=0.2, ent_coef=0.01,
        n_epochs=2, batch_size=16, value_clip_range=0.2,
    )
    _maybe_attach_split_attack_defend(cfg, trainer)
    if attach_teacher:
        from rl.custom_ppo.defend_teacher import DefendTeacherRunner

        trainer.defend_teacher_runner = DefendTeacherRunner(
            trainer.model, trainer.optimizers.primary, lambda_teacher=cfg.defend_teacher_lambda,
            cadence=1,
        )
    return trainer, env, cfg


def test_split_attack_defend_end_to_end_freezes_pi_a_and_trains_pi_d(tmp_path):
    """C-e/C-f: frozen pi_A's parameters, gradients, optimizer membership,
    and content hash are unchanged by a real collect+update cycle; pi_D
    receives real, nonzero parameter changes."""
    ckpt_path, frozen_params_at_save = _save_frozen_checkpoint(tmp_path, seed=101, name="a")
    trainer, env, cfg = _build_split_trainer(tmp_path, ckpt_path, seed=102, name="a")
    try:
        frozen_model = trainer.split_attack_defend_frozen_model
        assert frozen_model is not None
        assert bool(getattr(frozen_model, "role_conditioning_enabled", True)) is False
        for p in frozen_model.parameters():
            assert p.requires_grad is False

        for saved, loaded in zip(frozen_params_at_save, frozen_model.parameters()):
            assert torch.equal(saved, loaded.detach())
        hash_at_save = _state_dict_hash(frozen_model)

        # C-e (optimizer membership): the frozen model's parameters must
        # never appear in ANY of the trainer's optimizers' param_groups.
        frozen_param_ids = {id(p) for p in frozen_model.parameters()}
        for opt_name in ("primary", "actor", "critic", "router", "actor_cf"):
            opt = getattr(trainer.optimizers, opt_name, None)
            if opt is None:
                continue
            for group in opt.param_groups:
                for p in group["params"]:
                    assert id(p) not in frozen_param_ids, (
                        f"frozen pi_A parameter found in optimizers.{opt_name}"
                    )

        trainable_params_before = [p.detach().clone() for p in trainer.model.parameters()]

        rollout = trainer.collect_rollout()
        assert "defend_log_probs" in rollout.fields
        assert "obs_roles" in rollout.fields
        assert tuple(rollout.fields["defend_log_probs"].shape) == tuple(
            rollout.fields["log_probs"].shape
        )

        for before, after in zip(frozen_params_at_save, frozen_model.parameters()):
            assert torch.equal(before, after.detach()), "frozen model must not change during collection"
        assert all(p.grad is None for p in frozen_model.parameters()), (
            "frozen model must accumulate no gradient during collection (torch.no_grad())"
        )

        stats = trainer.update(rollout, total_timesteps=int(cfg.total_timesteps))
        assert stats is not None

        for before, after in zip(frozen_params_at_save, frozen_model.parameters()):
            assert torch.equal(before, after.detach()), "frozen pi_A must NEVER be touched by the optimizer"
        for p in frozen_model.parameters():
            assert p.grad is None or bool((p.grad == 0).all()), (
                "frozen pi_A must accumulate no nonzero gradient from a real update"
            )
        assert _state_dict_hash(frozen_model) == hash_at_save

        trainable_params_after = [p.detach().clone() for p in trainer.model.parameters()]
        changed = any(
            not torch.equal(a, b) for a, b in zip(trainable_params_before, trainable_params_after)
        )
        assert changed, "pi_D must receive gradient updates from the DEFEND-gated PPO loss"
        grads = [p.grad for p in trainer.model.parameters() if p.grad is not None]
        assert grads, "pi_D must have produced gradients during the update"
        assert any(bool((g != 0).any()) for g in grads), "pi_D's gradients must be nonzero on DEFEND samples"
    finally:
        env.close()


def test_split_attack_defend_old_log_prob_recompute_is_exact_before_any_update(tmp_path):
    """C-j: immediately after collection (before any parameter change),
    re-evaluating pi_D on the STORED executed actions and re-gating by the
    STORED role mask must reproduce the stored defend_log_probs exactly --
    the PPO ratio at the very first pass must be exp(0)=1 for every sample."""
    ckpt_path, _ = _save_frozen_checkpoint(tmp_path, seed=201, name="b")
    trainer, env, cfg = _build_split_trainer(tmp_path, ckpt_path, seed=202, name="b")
    try:
        rollout = trainer.collect_rollout()
        obs_batch = {
            "grid": rollout.fields["obs_grid"].reshape(-1, *rollout.fields["obs_grid"].shape[2:]),
            "vec": rollout.fields["obs_vec"].reshape(-1, *rollout.fields["obs_vec"].shape[2:]),
            "agent_mask": rollout.fields["obs_agent_mask"].reshape(-1, *rollout.fields["obs_agent_mask"].shape[2:]),
            "mask": rollout.fields["obs_mask"].reshape(-1, *rollout.fields["obs_mask"].shape[2:]),
        }
        roles_flat = rollout.fields["obs_roles"].reshape(-1, *rollout.fields["obs_roles"].shape[2:])
        actions_flat = rollout.fields["actions"].reshape(-1, *rollout.fields["actions"].shape[2:])
        gs_flat = rollout.fields["global_state"].reshape(-1, *rollout.fields["global_state"].shape[2:])
        stored_defend_lp = rollout.fields["defend_log_probs"].reshape(-1)

        with torch.no_grad():
            _values, _lp, _ent, aux = trainer.model.evaluate_actions(
                obs_batch, gs_flat, actions_flat, roles=roles_flat, return_per_agent=True,
            )
        from rl.custom_ppo.split_attack_defend import defend_gated_sum

        is_defend = roles_flat < 0.5
        recomputed = defend_gated_sum(aux["log_prob_per_agent"], is_defend)
        assert torch.allclose(recomputed, stored_defend_lp, atol=1e-5), (
            "re-evaluating pi_D on its own just-collected (obs, executed action) pairs must "
            "exactly reproduce the stored old log-prob before any parameter update"
        )
    finally:
        env.close()


def test_split_attack_defend_combines_with_defend_teacher_at_nonzero_lambda(tmp_path):
    """C-i: split-policy PPO functions correctly both with the DEFEND-teacher
    runner attached (lambda>0, the real training configuration) and without
    it (lambda=0 / absent, exercised by the other tests in this file)."""
    ckpt_path, _ = _save_frozen_checkpoint(tmp_path, seed=301, name="c")
    trainer, env, cfg = _build_split_trainer(
        tmp_path, ckpt_path, seed=302, name="c", attach_teacher=True
    )
    try:
        assert trainer.defend_teacher_runner is not None
        rollout = trainer.collect_rollout()
        stats = trainer.update(rollout, total_timesteps=int(cfg.total_timesteps))
        assert stats is not None
        assert trainer.defend_teacher_runner.n_ppo_actor_minibatches > 0
    finally:
        env.close()


def test_non_split_path_unchanged_when_flag_is_off(tmp_path):
    """C-k: with split_attack_defend_enabled left at its default (False),
    the rollout buffer carries no defend_log_probs field and the ordinary
    role-conditioned (or plain) PPO path runs exactly as it did before this
    spec existed."""
    from rl.custom_ppo.trainer import CustomPPOTrainer

    env = _build_env(seed=401)
    try:
        cfg = _base_cfg(run_tag="split_ad_off", checkpoint_dir=str(tmp_path))
        assert cfg.split_attack_defend_enabled is False
        trainer = CustomPPOTrainer(
            env, cfg, learning_rate=1e-3, clip_range=0.2, ent_coef=0.01,
            n_epochs=1, batch_size=16, value_clip_range=0.2,
        )
        rollout = trainer.collect_rollout()
        assert "defend_log_probs" not in rollout.fields
        stats = trainer.update(rollout, total_timesteps=int(cfg.total_timesteps))
        assert stats is not None
    finally:
        env.close()
