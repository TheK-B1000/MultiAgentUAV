from __future__ import annotations

import dataclasses

import torch

from experiments.run_exp2b_specialization_preserving_compression import build_exp2b_config
from experiments.run_exp2c_mode_specific_actor_compression import build_exp2c_config
from rl.custom_ppo import SharedActorCentralizedCritic
from rl.custom_ppo.exp2_teacher_compression import _shared_actor_parameters
from tests.test_exp2_k2_compression import _action_space, _obs, _obs_space


def _model(*, private_heads: bool) -> SharedActorCentralizedCritic:
    torch.manual_seed(11)
    return SharedActorCentralizedCritic(
        _obs_space(),
        _action_space(),
        latent_k=2,
        strategy_encoder_enabled=False,
        z_embed_dim=16,
        exp2c_mode_specific_action_heads=private_heads,
    )


def test_resolved_config_has_one_scientific_delta_vs_exp2b():
    parent, _ = build_exp2b_config()
    child, contract = build_exp2c_config()
    diff = {
        key for key, value in dataclasses.asdict(parent).items()
        if value != dataclasses.asdict(child)[key]
    }
    assert diff == {
        "checkpoint_dir", "episode_csv_path", "exp2_protocol_path",
        "exp2c_mode_specific_action_heads", "metrics_csv_path", "run_tag", "seed",
    }
    assert contract["scientific_diff_fields"] == ["exp2c_mode_specific_action_heads"]


def test_private_heads_are_exact_copies_with_no_other_private_capacity():
    model = _model(private_heads=True)
    actor = model.latent_actor
    assert actor.exp2c_mode_specific_action_heads is True
    assert actor.latent_adapters is None
    assert actor.latent_branch_trunks is None
    assert actor.latent_action_biases is None
    assert len(actor.latent_action_heads) == 2
    assert actor.latent_action_heads[0].weight.data_ptr() != actor.latent_action_heads[1].weight.data_ptr()
    for head in actor.latent_action_heads:
        assert torch.equal(head.weight, actor.action_head.weight)
        assert torch.equal(head.bias, actor.action_head.bias)


def test_z_rows_route_gradients_only_to_their_own_head_and_shared_body():
    model = _model(private_heads=True)
    obs = _obs(8)
    z0 = torch.zeros(8, dtype=torch.long)
    loss = model.policy_logits(obs, z_idx=z0).square().mean()
    grads = torch.autograd.grad(
        loss,
        [
            *model.latent_actor.body.parameters(),
            *model.latent_actor.latent_action_heads[0].parameters(),
            *model.latent_actor.latent_action_heads[1].parameters(),
        ],
        allow_unused=True,
    )
    n_body = len(list(model.latent_actor.body.parameters()))
    n_head = len(list(model.latent_actor.latent_action_heads[0].parameters()))
    assert any(g is not None and torch.count_nonzero(g) for g in grads[:n_body])
    assert all(g is not None and torch.count_nonzero(g) for g in grads[n_body:n_body + n_head])
    assert all(g is None or not torch.count_nonzero(g) for g in grads[n_body + n_head:])


def test_critic_is_architecturally_unchanged_and_checkpoint_roundtrip_is_exact():
    base = _model(private_heads=False)
    private = _model(private_heads=True)
    assert {
        name: tuple(value.shape) for name, value in base.critic.state_dict().items()
    } == {
        name: tuple(value.shape) for name, value in private.critic.state_dict().items()
    }
    clone = _model(private_heads=True)
    clone.load_state_dict(private.state_dict())
    obs = _obs(8)
    z = torch.tensor([0, 1] * 4, dtype=torch.long)
    assert torch.equal(private.policy_logits(obs, z_idx=z), clone.policy_logits(obs, z_idx=z))


def test_disabled_flag_preserves_the_existing_shared_head_path():
    model = _model(private_heads=False)
    assert model.latent_actor.latent_action_heads is None
    assert model.latent_actor.exp2c_mode_specific_action_heads is False


def test_shared_gradient_diagnostic_excludes_both_private_heads():
    model = _model(private_heads=True)
    shared_ids = {id(parameter) for parameter in _shared_actor_parameters(model)}
    private_ids = {
        id(parameter)
        for head in model.latent_actor.latent_action_heads
        for parameter in head.parameters()
    }
    assert shared_ids
    assert shared_ids.isdisjoint(private_ids)
