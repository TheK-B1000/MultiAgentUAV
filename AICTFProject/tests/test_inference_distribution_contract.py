from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

from game_field_gpu import VEC_OBS_DIM
from rl.custom_ppo import (
    CUSTOM_PPO_ACTOR_ARCH,
    CUSTOM_PPO_FORMAT,
    CUSTOM_PPO_LATENT_FORMAT,
    CUSTOM_PPO_VEC_SCHEMA_VERSION,
    CustomPPOInferencePolicy,
    SharedActorCentralizedCritic,
)
from rl.custom_ppo.distributions import MultiHeadActionDistribution
from rl.custom_ppo.inference import load_custom_ppo_checkpoint, load_custom_ppo_policy
from rl.custom_ppo.policy_contract import PolicyInferenceContract
from rl.latent_marl import CONTEXT_STATE_DIM


def _obs_space(*, channels: int = 7) -> spaces.Dict:
    return spaces.Dict(
        {
            "grid": spaces.Box(0.0, 1.0, shape=(2, channels, 20, 20), dtype=np.float32),
            "vec": spaces.Box(-1.0, 1.0, shape=(2, VEC_OBS_DIM), dtype=np.float32),
            "agent_mask": spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32),
            "mask": spaces.Box(0.0, 1.0, shape=(110,), dtype=np.float32),
        }
    )


def _action_space() -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete([5, 50, 5, 50])


def _obs_tensors(*, batch: int = 2, channels: int = 7) -> dict[str, torch.Tensor]:
    return {
        "grid": torch.linspace(
            0.0,
            1.0,
            steps=batch * 2 * channels * 20 * 20,
            dtype=torch.float32,
        ).reshape(batch, 2, channels, 20, 20),
        "vec": torch.zeros((batch, 2, VEC_OBS_DIM), dtype=torch.float32),
        "agent_mask": torch.ones((batch, 2), dtype=torch.float32),
        "mask": torch.ones((batch, 110), dtype=torch.float32),
    }


def _write_checkpoint(path: Path, *, latent: bool) -> SharedActorCentralizedCritic:
    torch.manual_seed(20260627 if latent else 20260628)
    model = SharedActorCentralizedCritic(
        _obs_space(),
        _action_space(),
        actor_cnn_feature_dim=128,
        latent_k=4 if latent else 0,
        z_embed_dim=16,
        strategy_hidden_dim=128,
        critic_hidden_dim=128,
    )
    cfg = {
        "seed": 7,
        "max_blue_agents": 2,
        "use_latent_strategy": latent,
        "latent_k": 4 if latent else 0,
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
            "format": CUSTOM_PPO_LATENT_FORMAT if latent else CUSTOM_PPO_FORMAT,
            "actor_arch": CUSTOM_PPO_ACTOR_ARCH,
            "actor_cnn_feature_dim": 128,
            "global_state_dim": CONTEXT_STATE_DIM if latent else 34,
            "vec_schema_version": CUSTOM_PPO_VEC_SCHEMA_VERSION,
        },
        path,
    )
    return model


class LoadedPolicyDistributionContractTests(unittest.TestCase):
    def test_loaded_latent_policy_and_model_expose_distribution_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "latent.zip"
            _write_checkpoint(ckpt, latent=True)

            loaded = load_custom_ppo_checkpoint(
                str(ckpt), _obs_space(), _action_space(), device="cpu"
            )

            self.assertIsInstance(loaded.policy, CustomPPOInferencePolicy)
            self.assertIsInstance(loaded.policy, PolicyInferenceContract)
            self.assertIsInstance(loaded.policy.model, PolicyInferenceContract)
            self.assertTrue(callable(loaded.policy.get_distribution))
            self.assertTrue(callable(loaded.policy.model.get_distribution))

    def test_loaded_wrapper_preserves_gradients_for_explicit_latent_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "latent.zip"
            _write_checkpoint(ckpt, latent=True)
            policy = load_custom_ppo_policy(str(ckpt), _obs_space(), _action_space(), device="cpu")

            obs = _obs_tensors(batch=2)
            obs["grid"].requires_grad_(True)
            z_idx = torch.zeros((2,), dtype=torch.long)

            dist = policy.get_distribution(obs, z_idx=z_idx)
            self.assertIsInstance(dist, MultiHeadActionDistribution)

            loss = sum(head.logits.square().mean() for head in dist.heads)
            loss.backward()

            self.assertIsNotNone(obs["grid"].grad)
            self.assertGreater(float(obs["grid"].grad.abs().sum().item()), 0.0)

    def test_loaded_latent_model_requires_explicit_latent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "latent.zip"
            _write_checkpoint(ckpt, latent=True)
            policy = load_custom_ppo_policy(str(ckpt), _obs_space(), _action_space(), device="cpu")

            with self.assertRaisesRegex(ValueError, "z_idx"):
                policy.model.get_distribution(_obs_tensors(batch=1))

    def test_loaded_non_latent_policy_does_not_require_latent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "non_latent.zip"
            _write_checkpoint(ckpt, latent=False)
            policy = load_custom_ppo_policy(str(ckpt), _obs_space(), _action_space(), device="cpu")

            dist = policy.get_distribution(_obs_tensors(batch=1))
            self.assertIsInstance(dist, MultiHeadActionDistribution)

    def test_evaluator_preflight_rejects_missing_distribution_contract(self) -> None:
        from rl.evaluation.preflight import validate_distribution_contract

        class BrokenPolicy:
            model = object()

        with self.assertRaises(TypeError):
            validate_distribution_contract(BrokenPolicy(), label="broken")


if __name__ == "__main__":
    unittest.main()
