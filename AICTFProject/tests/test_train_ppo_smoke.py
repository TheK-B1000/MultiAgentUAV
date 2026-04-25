"""End-to-end smoke: GPU batched env -> local PPO update path."""

from __future__ import annotations

import os

# `train_ppo` pulls TensorBoard/TensorFlow; set before importing (unittest may not load `tests` package first).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import unittest
from pathlib import Path

from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
from rl.custom_ppo import load_custom_ppo_policy, read_custom_ppo_metadata
from rl.train_ppo import PPOConfig, train_ppo


_WORKSPACE_TMP = Path(__file__).resolve().parents[1] / ".test_runs" / "train_ppo_smoke"


def _smoke_ppo_config(*, run_tag: str, checkpoint_dir: str) -> PPOConfig:
    cfg = PPOConfig()
    cfg.seed = 0
    cfg.total_timesteps = 8
    cfg.n_envs = 1
    cfg.n_steps = 8
    cfg.batch_size = 8
    cfg.n_epochs = 1
    # Avoid stable-MARL override (n_epochs=2, large default batch) so the smoke stays tiny and predictable.
    cfg.use_stable_marl_ppo = False
    cfg.device = "cpu"
    cfg.enable_progress_bar = False
    cfg.enable_tensorboard = False
    cfg.enable_checkpoints = False
    cfg.enable_eval = False
    cfg.verbose_training = False
    cfg.max_blue_agents = 2
    cfg.mode = "FIXED_OPPONENT"
    cfg.fixed_opponent_tag = "OP3"
    cfg.gpu_native_env = True
    cfg.run_tag = run_tag
    cfg.checkpoint_dir = checkpoint_dir
    return cfg


def _run_smoke_and_cleanup(*, tag: str) -> None:
    _WORKSPACE_TMP.mkdir(parents=True, exist_ok=True)
    cfg = _smoke_ppo_config(
        run_tag=tag,
        checkpoint_dir=str(_WORKSPACE_TMP),
    )
    final_zip = _WORKSPACE_TMP / f"final_{tag}.zip"
    try:
        train_ppo(cfg)
        assert final_zip.is_file(), f"expected {final_zip}"
    finally:
        if final_zip.exists():
            final_zip.unlink()


class TrainPpoSmokeTests(unittest.TestCase):
    def test_train_ppo_smoke_custom_few_steps(self) -> None:
        _run_smoke_and_cleanup(tag="unittest_smoke_custom_ppo_2v2")

    def test_latent_training_is_reserved_for_followup_phase(self) -> None:
        cfg = _smoke_ppo_config(run_tag="unittest_reserved_latent_2v2", checkpoint_dir=str(_WORKSPACE_TMP))
        cfg.use_latent_strategy = True
        with self.assertRaises(NotImplementedError):
            train_ppo(cfg)

    def test_saved_checkpoint_loads_for_local_inference(self) -> None:
        _WORKSPACE_TMP.mkdir(parents=True, exist_ok=True)
        tag = "unittest_inference_custom_ppo_2v2"
        cfg = _smoke_ppo_config(run_tag=tag, checkpoint_dir=str(_WORKSPACE_TMP))
        final_zip = _WORKSPACE_TMP / f"final_{tag}.zip"
        env = None
        try:
            train_ppo(cfg)
            meta = read_custom_ppo_metadata(str(final_zip))
            self.assertEqual(meta["n_blue"], 2)
            env = GPUCTFVecEnv(GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu", seed=123))
            obs = env.reset()
            policy = load_custom_ppo_policy(str(final_zip), env.observation_space, env.action_space, device="cpu")
            actions, _ = policy.predict(obs, deterministic=True)
            self.assertEqual(actions.shape, (4,))
        finally:
            if env is not None:
                env.close()
            if final_zip.exists():
                final_zip.unlink()


if __name__ == "__main__":
    unittest.main()
