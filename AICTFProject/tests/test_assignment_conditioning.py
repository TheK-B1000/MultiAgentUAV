"""Executable contracts for ASSIGNMENT_CONDITIONING_V1 (exploratory / diagnostic)."""

from __future__ import annotations

import pytest
import torch
from gymnasium.spaces import Box, Dict, MultiDiscrete

from rl.custom_ppo.guard_assignment import (
    encode_assignment_features,
    assignment_from_core,
    AssignmentHoldState,
)
from rl.custom_ppo.policy import SharedActorCentralizedCritic


def _spaces(n_agents: int = 4):
    grid = Box(low=0, high=1, shape=(n_agents, 8, 11, 11), dtype="float32")
    vec = Box(low=-1, high=1, shape=(n_agents, 20), dtype="float32")
    obs = Dict(
        grid=grid,
        vec=vec,
        agent_mask=Box(low=0, high=1, shape=(n_agents,), dtype="float32"),
        mask=Box(low=0, high=1, shape=(16,), dtype="float32"),
    )
    act = MultiDiscrete([5, 5] * n_agents)
    return obs, act


def test_encode_assignment_features_shape_and_range():
    B, N, Ne = 2, 4, 4
    resp = torch.tensor([[0, 1, 2, 0], [0, 0, 1, 2]])
    ent = torch.tensor([[-2, -1, 0, 1], [1, 2, -2, 0]])
    dx = torch.randn(B, N)
    dy = torch.randn(B, N)
    feat = encode_assignment_features(resp, ent, dx, dy, defense_radius=6.0, n_enemies=Ne)
    assert feat.shape == (B, N, 4)
    assert feat.dtype == torch.float32
    assert torch.all(feat[..., 0] >= 0.0) and torch.all(feat[..., 0] <= 1.0)
    assert torch.all(feat[..., 1] >= 0.0) and torch.all(feat[..., 1] <= 1.0)
    assert torch.all(feat[..., 2].abs() <= 1.0 + 1e-6)
    assert torch.all(feat[..., 3].abs() <= 1.0 + 1e-6)


def test_mutual_exclusion_role_and_assignment_at_policy_construct():
    obs_s, act_s = _spaces()
    SharedActorCentralizedCritic(
        obs_s, act_s, strategy_encoder_enabled=False, latent_k=0,
        role_conditioning_enabled=True,
    )
    SharedActorCentralizedCritic(
        obs_s, act_s, strategy_encoder_enabled=False, latent_k=0,
        assignment_conditioning_enabled=True,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        SharedActorCentralizedCritic(
            obs_s, act_s, strategy_encoder_enabled=False, latent_k=0,
            role_conditioning_enabled=True,
            assignment_conditioning_enabled=True,
        )


def test_g1_style_assignment_from_core_does_not_crash():
    from game_field_gpu import BatchedCTFCore, GPUFieldConfig

    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=4,
        max_red_agents=4,
        device="cpu",
        max_decision_steps=32,
        map_layout="map_a_open",
    )
    core = BatchedCTFCore(cfg)
    core.reset_all()
    hold = AssignmentHoldState(1, 4, hold_ticks=8, n_enemies=4, device="cpu")
    feat = assignment_from_core(core, hold, force=True)
    assert feat.shape == (1, 4, 4)
    assert torch.isfinite(feat).all()
