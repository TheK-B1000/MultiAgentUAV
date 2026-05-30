from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch.distributions import Categorical

from game_field_gpu import VEC_OBS_DIM
from rl.custom_ppo import (
    CUSTOM_PPO_ACTOR_ARCH,
    CUSTOM_PPO_LATENT_FORMAT,
    CUSTOM_PPO_VEC_SCHEMA_VERSION,
    SharedActorCentralizedCritic,
    load_custom_ppo_policy,
    read_custom_ppo_metadata,
)
from rl.latent_marl import CONTEXT_STATE_DIM


def _obs_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "grid": spaces.Box(low=0.0, high=1.0, shape=(2, 7, 20, 20), dtype=np.float32),
            "vec": spaces.Box(low=-1.0, high=1.0, shape=(2, VEC_OBS_DIM), dtype=np.float32),
            "agent_mask": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            "mask": spaces.Box(low=0.0, high=1.0, shape=(110,), dtype=np.float32),
        }
    )


def _action_space() -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete([5, 50, 5, 50])


def _fixed_obs(batch: int = 3) -> dict[str, torch.Tensor]:
    grid = torch.linspace(0.0, 1.0, steps=batch * 2 * 7 * 20 * 20, dtype=torch.float32).reshape(
        batch, 2, 7, 20, 20
    )
    vec = torch.linspace(-0.75, 0.75, steps=batch * 2 * VEC_OBS_DIM, dtype=torch.float32).reshape(
        batch, 2, VEC_OBS_DIM
    )
    mask = torch.ones((batch, 110), dtype=torch.float32)
    mask[1, 7] = 0.0
    mask[2, 63] = 0.0
    return {
        "grid": grid,
        "vec": vec,
        "agent_mask": torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        "mask": mask,
    }


def _fixed_global_state(batch: int = 3) -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, steps=batch * CONTEXT_STATE_DIM, dtype=torch.float32).reshape(
        batch, CONTEXT_STATE_DIM
    )


def _fixed_actions() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 3, 4, 49],
            [2, 17, 1, 9],
            [4, 0, 0, 31],
        ],
        dtype=torch.long,
    )


def _ref_policy_logits(
    model: SharedActorCentralizedCritic,
    obs: dict[str, torch.Tensor],
    z_idx: torch.Tensor,
) -> torch.Tensor:
    grid = obs["grid"].float()
    vec = obs["vec"].float()
    batch = int(grid.shape[0])
    cnn_features = model.actor_cnn(grid.reshape(batch * model.n_agents, *model.grid_shape))
    cnn_features = cnn_features.reshape(batch, model.n_agents, model.actor_cnn_feature_dim)
    vloc = vec.float()
    agent_mask = obs.get("agent_mask")
    if agent_mask is not None:
        if agent_mask.dim() == 1:
            agent_mask = agent_mask.unsqueeze(0)
        mask = agent_mask.float().unsqueeze(-1)
        cnn_features = cnn_features * mask
        vloc = vloc * mask
    assert model.strategy_embedding is not None
    z = z_idx.long().reshape(-1).clamp(min=0, max=model.latent_k - 1)
    z_emb = model.strategy_embedding(z).unsqueeze(1).expand(batch, model.n_agents, model.z_embed_dim)
    actor_in = torch.cat([cnn_features, vloc, z_emb], dim=-1)
    hidden = model.actor_body(actor_in.reshape(batch * model.n_agents, -1))
    return model.actor_head(hidden).reshape(batch, model.n_agents * model.per_agent_logits)


def _ref_critic_extra(model: SharedActorCentralizedCritic, actions: torch.Tensor, z_idx: torch.Tensor) -> torch.Tensor:
    chunks = []
    for col, dim in enumerate(model.action_dims):
        action = actions[:, col].long().clamp(min=0, max=int(dim) - 1)
        chunks.append(F.one_hot(action, num_classes=int(dim)).float())
    z = z_idx.long().reshape(-1).clamp(min=0, max=model.latent_k - 1)
    chunks.append(F.one_hot(z, num_classes=model.latent_k).float())
    return torch.cat(chunks, dim=-1)


def _ref_values(
    model: SharedActorCentralizedCritic,
    global_state: torch.Tensor,
    actions: torch.Tensor,
    z_idx: torch.Tensor,
) -> torch.Tensor:
    return model.critic(global_state.float(), extra=_ref_critic_extra(model, actions, z_idx)).squeeze(-1)


def _write_fixed_checkpoint(path: Path) -> SharedActorCentralizedCritic:
    torch.manual_seed(1234)
    model = SharedActorCentralizedCritic(
        _obs_space(),
        _action_space(),
        actor_cnn_feature_dim=128,
        latent_k=4,
        z_embed_dim=16,
        strategy_hidden_dim=128,
        critic_hidden_dim=128,
    )
    model.eval()
    cfg = {
        "seed": 1234,
        "max_blue_agents": 2,
        "use_latent_strategy": True,
        "latent_k": 4,
        "latent_z_embed_dim": 16,
        "latent_strategy_hidden": 128,
        "latent_vf_hidden": 128,
        "latent_strategy_aux_return_head": False,
        "latent_episode_strategy_ppo": False,
        "actor_cnn_feature_dim": 128,
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": cfg,
            "format": CUSTOM_PPO_LATENT_FORMAT,
            "actor_arch": CUSTOM_PPO_ACTOR_ARCH,
            "actor_cnn_feature_dim": 128,
            "global_state_dim": CONTEXT_STATE_DIM,
            "vec_schema_version": CUSTOM_PPO_VEC_SCHEMA_VERSION,
        },
        path,
    )
    return model


class CustomPpoPolicyParityTests(unittest.TestCase):
    def test_loaded_latent_policy_matches_decomposed_reference_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "fixed_latent_policy.zip"
            original_model = _write_fixed_checkpoint(checkpoint)

            metadata = read_custom_ppo_metadata(str(checkpoint))
            self.assertEqual(metadata["format"], CUSTOM_PPO_LATENT_FORMAT)
            self.assertTrue(metadata["use_latent_strategy"])
            self.assertEqual(metadata["latent_k"], 4)

            loaded_policy = load_custom_ppo_policy(
                str(checkpoint),
                _obs_space(),
                _action_space(),
                device="cpu",
            )
            loaded_model = loaded_policy.model
            loaded_model.eval()

            obs = _fixed_obs()
            global_state = _fixed_global_state()
            z_idx = torch.tensor([0, 2, 3], dtype=torch.long)
            actions = _fixed_actions()

            with torch.no_grad():
                original_logits = original_model.policy_logits(obs, z_idx=z_idx)
                production_logits = loaded_model.policy_logits(obs, z_idx=z_idx)
                reference_logits = _ref_policy_logits(loaded_model, obs, z_idx)

                production_masked = loaded_model._mask_logits(production_logits, obs["mask"])
                reference_masked = loaded_model._mask_logits(reference_logits, obs["mask"])

                original_values = original_model.values(global_state, actions=actions, z_idx=z_idx)
                production_values = loaded_model.values(global_state, actions=actions, z_idx=z_idx)
                reference_values = _ref_values(loaded_model, global_state, actions, z_idx)

                production_z, production_z_log_prob, production_z_entropy, production_z_logits = (
                    loaded_model.sample_strategy(global_state, deterministic=True)
                )
                reference_z_logits = loaded_model.strategy_encoder(global_state.float())
                reference_dist = Categorical(logits=reference_z_logits)
                reference_z = torch.argmax(reference_z_logits, dim=-1)
                reference_z_log_prob = reference_dist.log_prob(reference_z)
                reference_z_entropy = reference_dist.entropy()

            torch.testing.assert_close(production_logits, original_logits, rtol=0.0, atol=0.0)
            torch.testing.assert_close(production_logits, reference_logits, rtol=0.0, atol=0.0)
            torch.testing.assert_close(production_masked, reference_masked, rtol=0.0, atol=0.0)
            torch.testing.assert_close(production_values, original_values, rtol=0.0, atol=0.0)
            torch.testing.assert_close(production_values, reference_values, rtol=0.0, atol=0.0)
            torch.testing.assert_close(production_z_logits, reference_z_logits, rtol=0.0, atol=0.0)
            torch.testing.assert_close(production_z, reference_z, rtol=0.0, atol=0.0)
            torch.testing.assert_close(production_z_log_prob, reference_z_log_prob, rtol=0.0, atol=0.0)
            torch.testing.assert_close(production_z_entropy, reference_z_entropy, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
