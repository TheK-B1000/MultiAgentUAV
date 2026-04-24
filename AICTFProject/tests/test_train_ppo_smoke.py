"""End-to-end smoke: GPU batched env → VecEnv (optional latent) → PPO update path."""

from __future__ import annotations

import os

# `train_ppo` pulls TensorBoard/TensorFlow; set before importing (unittest may not load `tests` package first).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import unittest
from pathlib import Path

from rl.train_ppo import PPOConfig, train_ppo


_WORKSPACE_TMP = Path(__file__).resolve().parents[1] / "checkpoints_sb3" / "2v2"


def _smoke_ppo_config(*, use_latent_strategy: bool, run_tag: str, checkpoint_dir: str) -> PPOConfig:
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
    cfg.use_latent_strategy = use_latent_strategy
    if use_latent_strategy:
        cfg.latent_k = 4
        cfg.latent_resample_every_n = 0
    return cfg


def _run_smoke_and_cleanup(*, use_latent_strategy: bool, tag: str) -> None:
    _WORKSPACE_TMP.mkdir(parents=True, exist_ok=True)
    cfg = _smoke_ppo_config(
        use_latent_strategy=use_latent_strategy,
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
    def test_train_ppo_smoke_vanilla_ppo_few_steps(self) -> None:
        _run_smoke_and_cleanup(
            use_latent_strategy=False,
            tag="unittest_smoke_ppo_cnn_2v2",
        )

    def test_train_ppo_smoke_latent_strategy_few_steps(self) -> None:
        _run_smoke_and_cleanup(
            use_latent_strategy=True,
            tag="unittest_smoke_latent_2v2",
        )


if __name__ == "__main__":
    unittest.main()
