from __future__ import annotations

import torch
import torch.nn as nn
import pytest
import numpy as np
from gymnasium import spaces

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.exp2_teacher_compression import directed_identity_kl
from rl.custom_ppo.policy import SharedActorCentralizedCritic
from rl.latent_marl import CONTEXT_STATE_DIM
from rl.networks import CentralizedCritic
from rl.scorer.qpsi import QPsi, QPsiConfig


def _valid_vec(batch: int) -> torch.Tensor:
    vec = torch.zeros(batch, 2, 20)
    vec[..., 6] = 2.0 / 20.0
    vec[..., 7] = 10.0 / 20.0
    return vec


def test_qpsi_single_regime_keeps_legacy_shape_and_names():
    model = QPsi(QPsiConfig(hidden=16, conv_width=8, action_dim=4, rank=2))
    assert hasattr(model, "head_b")
    assert not hasattr(model, "regime_heads")
    grid = torch.zeros(3, 2, 7, 20, 20)
    vec = _valid_vec(3)
    agent_mask = torch.ones(3, 2)
    pole = torch.tensor([0, 1, 0])
    result = model(grid, vec, agent_mask, pole, torch.zeros(3, dtype=torch.long),
                   torch.ones(3, dtype=torch.long))
    assert result.shape == (3,)


def test_regime_from_vec_covers_all_four_d1_cells():
    model = QPsi(QPsiConfig(n_regimes=4, hidden=16, conv_width=8, action_dim=4, rank=2))
    vec = _valid_vec(4)
    vec[1, 0, 10] = 1.0
    vec[2:, :, 6] = 5.0 / 20.0
    vec[2:, :, 7] = 5.0 / 20.0
    vec[3, 1, 10] = 1.0
    assert model.regime_from_vec(vec).tolist() == [0, 1, 2, 3]


def test_regime_reconstruction_fails_closed_on_schema_and_disagreement():
    model = QPsi(QPsiConfig(n_regimes=4, hidden=16, conv_width=8, action_dim=4, rank=2))
    with pytest.raises(ValueError, match="shape"):
        model.regime_from_vec(torch.zeros(2, 20))
    vec = _valid_vec(1)
    vec[0, 1, 6] += 0.1
    with pytest.raises(ValueError, match="disagrees"):
        model.regime_from_vec(vec)


def test_private_critic_starts_function_equivalent_to_shared():
    torch.manual_seed(17)
    shared = CentralizedCritic(global_state_dim=3, hidden_dim=8, extra_dim=2)
    torch.manual_seed(17)
    private = CentralizedCritic(
        global_state_dim=3, hidden_dim=8, extra_dim=2, private_z_heads=True
    )
    private.copy_shared_head_into_private()
    state = torch.randn(5, 3)
    for z in (0, 1):
        extra = torch.nn.functional.one_hot(
            torch.full((5,), z), num_classes=2
        ).float()
        assert torch.equal(private(state, extra), shared(state, extra))


def test_private_critic_routes_gradients_to_selected_head_only():
    critic = CentralizedCritic(
        global_state_dim=3, hidden_dim=8, extra_dim=2, private_z_heads=True
    )
    critic.copy_shared_head_into_private()
    state = torch.randn(4, 3)
    z0 = torch.tensor([[1.0, 0.0]]).repeat(4, 1)
    critic(state, z0).sum().backward()
    assert critic.head_V0.weight.grad is not None
    assert critic.head_V0.weight.grad.abs().sum() > 0
    assert critic.head_V1.weight.grad is None
    assert critic.head_V1.bias.grad is None


def test_legacy_shared_critic_checkpoint_maps_into_both_private_heads():
    observation_space = spaces.Dict({
        "grid": spaces.Box(0.0, 1.0, shape=(2, 7, 20, 20), dtype=np.float32),
        "vec": spaces.Box(-1.0, 1.0, shape=(2, 20), dtype=np.float32),
        "agent_mask": spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32),
        "mask": spaces.Box(0.0, 1.0, shape=(110,), dtype=np.float32),
    })
    action_space = spaces.MultiDiscrete([5, 50, 5, 50])
    torch.manual_seed(23)
    shared = SharedActorCentralizedCritic(
        observation_space, action_space, latent_k=2, strategy_encoder_enabled=False
    )
    torch.manual_seed(99)
    private = SharedActorCentralizedCritic(
        observation_space,
        action_space,
        latent_k=2,
        strategy_encoder_enabled=False,
        rasr_private_critic_heads=True,
    )
    private.load_state_dict(shared.state_dict(), strict=True)
    state = torch.randn(4, CONTEXT_STATE_DIM)
    for z in (0, 1):
        extra = torch.nn.functional.one_hot(torch.full((4,), z), 2).float()
        assert torch.equal(private.critic(state, extra), shared.critic(state, extra))


class _ToyPolicy(nn.Module):
    action_dims = (2, 2)
    n_agents = 2

    def __init__(self, *, teacher_logits: tuple[float, float] | None = None):
        super().__init__()
        if teacher_logits is None:
            self.mode_logits = nn.Parameter(torch.zeros(2, 2, 2))
            self.teacher_logits = None
        else:
            self.mode_logits = None
            self.teacher_logits = nn.Parameter(
                torch.tensor(teacher_logits, dtype=torch.float32).repeat(2, 1),
            )

    def policy_logits(self, obs, *, z_idx=None):
        batch = obs["mask"].shape[0]
        if self.mode_logits is None:
            return self.teacher_logits.reshape(1, -1).repeat(batch, 1)
        return self.mode_logits.index_select(0, z_idx).reshape(batch, -1)

    @staticmethod
    def _mask_logits(logits, mask):
        return logits.masked_fill(mask <= 0, -1e9)


def test_directed_identity_is_finite_and_only_students_receive_gradients():
    student = _ToyPolicy()
    teachers = {
        0: _ToyPolicy(teacher_logits=(3.0, -3.0)),
        1: _ToyPolicy(teacher_logits=(-3.0, 3.0)),
    }
    obs = {"mask": torch.ones(8, 4)}
    z = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    decision = torch.ones(8, 2, dtype=torch.bool)
    loss, metrics = directed_identity_kl(student, teachers, obs, z, decision)
    assert torch.isfinite(loss)
    loss.backward()
    assert student.mode_logits.grad is not None
    assert student.mode_logits.grad.abs().sum() > 0
    assert all(parameter.grad is None for teacher in teachers.values()
               for parameter in teacher.parameters())
    assert {"identity_gap_A", "identity_gap_B"} <= set(metrics)


def test_rasr_defaults_are_structurally_off():
    cfg = PPOConfig()
    assert cfg.rasr_regime_qpsi is False
    assert cfg.rasr_regime_qpsi_sha256 == (
        "44c0680e037939de287ad4201fead6312bc92b6bcd1fd902f568868cb24b760a"
    )
    assert cfg.rasr_private_critic_heads is False
    assert cfg.rasr_directed_identity is False
    critic = CentralizedCritic(global_state_dim=3, hidden_dim=8, extra_dim=2)
    assert critic.private_z_heads is False
    assert len(critic.net) == 5
