"""E3-related verification: no-latent never touches the strategy path; inference replay; E3 CSV."""

from __future__ import annotations

import csv
import os
import unittest
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from rl.custom_ppo import E3_STEP_TELEMETRY_FIELDS, SharedActorCentralizedCritic, load_custom_ppo_policy
from rl.train_ppo import PPOConfig, train_ppo

_WORK = Path(__file__).resolve().parents[1] / ".test_runs" / "e3_verify"


def _cleanup(tag: str) -> None:
    for suffix in (".zip", "_metrics.csv", "_episodes.csv"):
        path = _WORK / (f"final_{tag}{suffix}" if suffix == ".zip" else f"{tag}{suffix}")
        if path.is_file():
            path.unlink()
    e3p = _WORK / f"{tag}_e3.csv"
    if e3p.is_file():
        e3p.unlink()


def _base_cfg(*, run_tag: str) -> PPOConfig:
    cfg = PPOConfig()
    cfg.seed = 7
    cfg.total_timesteps = 8
    cfg.n_envs = 1
    cfg.n_steps = 8
    cfg.batch_size = 8
    cfg.n_epochs = 1
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
    cfg.checkpoint_dir = str(_WORK)
    return cfg


def _load_policy_from_zip(path: Path):
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig

    c = GPUFieldConfig()
    c.max_blue = 2
    c.max_red = 2
    c.batch = 1
    env = GPUCTFVecEnv(c)
    p = load_custom_ppo_policy(str(path), env.observation_space, env.action_space, device="cpu")
    return p, env


class E3RngVerificationTests(unittest.TestCase):
    def test_no_latent_never_invokes_sample_strategy(self) -> None:
        sample_calls: list[int] = []
        _orig = SharedActorCentralizedCritic.sample_strategy

        def _track(self, *a, **k):  # type: ignore[no-untyped-def]
            sample_calls.append(1)
            return _orig(self, *a, **k)

        SharedActorCentralizedCritic.sample_strategy = _track  # type: ignore[assignment]
        _WORK.mkdir(parents=True, exist_ok=True)
        tag = "unittest_e3_no_latent_sample_guard"
        cfg = _base_cfg(run_tag=tag)
        cfg.use_latent_strategy = False
        try:
            train_ppo(cfg)
        finally:
            SharedActorCentralizedCritic.sample_strategy = _orig  # type: ignore[assignment]
        self.assertEqual(
            len(sample_calls),
            0,
            "use_latent_strategy=False must never call sample_strategy (strategy RNG untouched).",
        )
        _cleanup(tag)

    def test_latent_stochastic_session_repeats_on_reload(self) -> None:
        _WORK.mkdir(parents=True, exist_ok=True)
        tag = "unittest_e3_latent_replay"
        cfg = _base_cfg(run_tag=tag)
        cfg.use_latent_strategy = True
        cfg.latent_resample_every_n = 0
        zfile = _WORK / f"final_{tag}.zip"
        try:
            train_ppo(cfg)
            self.assertTrue(zfile.is_file(), f"missing {zfile}")

            def run_session():
                p, env = _load_policy_from_zip(zfile)
                p.reset_strategy()
                obs = env.reset()
                return [p.predict(obs, deterministic=False)[0].copy() for _ in range(3)]

            s1 = run_session()
            s2 = run_session()
            for i in range(3):
                self.assertTrue(
                    (s1[i] == s2[i]).all(),
                    f"inference session replay mismatch at i={i}",
                )
        finally:
            _cleanup(tag)

    def test_e3_telemetry_csv_columns_and_header(self) -> None:
        _WORK.mkdir(parents=True, exist_ok=True)
        tag = "unittest_e3_csv"
        cfg = _base_cfg(run_tag=tag)
        cfg.use_latent_strategy = True
        cfg.latent_resample_every_n = 0
        e3_path = _WORK / f"{tag}_e3.csv"
        cfg.e3_step_telemetry_path = str(e3_path)
        try:
            train_ppo(cfg)
            self.assertTrue(e3_path.is_file(), "e3 step telemetry file should be created when path set + latent on")
            with e3_path.open(newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                self.assertEqual(tuple(r.fieldnames or ()), E3_STEP_TELEMETRY_FIELDS)
        finally:
            _cleanup(tag)


if __name__ == "__main__":
    unittest.main()
