"""End-to-end smoke: GPU batched env -> local PPO update path."""

from __future__ import annotations

import os
import csv

# `train_ppo` pulls TensorBoard/TensorFlow; set before importing (unittest may not load `tests` package first).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import unittest
from pathlib import Path

import torch

from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
from plot.compare_reward_updates import COMPARISON_COLUMNS, compare_policy_updates, format_markdown_table
from rl.custom_ppo import (
    CUSTOM_PPO_ACTOR_ARCH,
    CUSTOM_PPO_FORMAT,
    CUSTOM_PPO_LATENT_FORMAT,
    CUSTOM_PPO_VEC_SCHEMA_VERSION,
    CustomPPOTrainer,
    SharedActorCentralizedCritic,
    _METRICS_CSV_LEGACY_COLUMN_FILL,
    load_custom_ppo_policy,
    read_custom_ppo_metadata,
)
from rl.custom_ppo.csv_writers import _write_csv_row
from rl.train_ppo import (
    PPOConfig,
    TrainMode,
    _acquire_run_lock,
    _apply_training_preset,
    _gpu_env_reward_kwargs,
    _strip_eval_only_opponents_from_training_pool,
    train_ppo,
)


_WORKSPACE_TMP = Path(__file__).resolve().parents[1] / ".test_runs" / "train_ppo_smoke"


def _load_checkpoint_payload(path: Path) -> dict:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _cleanup_training_outputs(tag: str) -> None:
    for suffix in (".zip", "_metrics.csv", "_episodes.csv", "_strategy_experience.csv"):
        path = _WORKSPACE_TMP / (f"final_{tag}{suffix}" if suffix == ".zip" else f"{tag}{suffix}")
        if path.exists():
            path.unlink()


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
        _cleanup_training_outputs(tag)


class TrainPpoSmokeTests(unittest.TestCase):
    def test_push80_preset_applies_expected_knobs(self) -> None:
        cfg = _apply_training_preset(PPOConfig(), "latent_op3_push80_1m")
        self.assertTrue(cfg.use_latent_strategy)
        self.assertEqual(cfg.fixed_opponent_tag, "OP3")
        self.assertTrue(cfg.normalize_returns)
        self.assertEqual(cfg.latent_entropy_objective, "minimize")
        self.assertAlmostEqual(cfg.latent_lam_h, 0.01)
        self.assertAlmostEqual(cfg.latent_lam_p, 0.04)
        self.assertEqual(cfg.n_epochs, 8)
        self.assertEqual(cfg.batch_size, 512)

    def test_train80_preset_applies_expected_knobs(self) -> None:
        cfg = _apply_training_preset(PPOConfig(), "latent_train80_op3_1m")
        self.assertTrue(cfg.use_latent_strategy)
        self.assertEqual(cfg.fixed_opponent_tag, "OP3")
        self.assertEqual(cfg.mode, "FIXED_OPPONENT")
        self.assertTrue(cfg.normalize_returns)
        self.assertEqual(cfg.latent_entropy_objective, "minimize")
        self.assertAlmostEqual(cfg.ent_coef, 0.001)
        self.assertAlmostEqual(cfg.latent_lam_h, 0.02)
        self.assertAlmostEqual(cfg.latent_lam_p, 0.06)
        self.assertEqual(cfg.n_epochs, 10)
        self.assertEqual(cfg.batch_size, 512)
        self.assertTrue(cfg.latent_strategy_aux_return_head)
        self.assertAlmostEqual(cfg.latent_strategy_aux_return_coef, 0.75)

    def test_wrmax_preset_applies_expected_knobs(self) -> None:
        for preset in ("latent_op3_wrmax_2m", "latent_op3_wrmax_1m"):
            cfg = _apply_training_preset(PPOConfig(), preset)
            self.assertEqual(cfg.total_timesteps, 1_000_000, msg=preset)
            self.assertEqual(cfg.run_tag, "latent_op3_wrmax_1m_2v2", msg=preset)
            self.assertTrue(cfg.latent_strategy_aux_return_head)
            self.assertEqual(cfg.latent_entropy_objective, "minimize")
            self.assertAlmostEqual(cfg.vf_coef, 1.1)
            self.assertEqual(cfg.latent_resample_every_n, 0)
            self.assertEqual(cfg.latent_vf_hidden, 256)
            self.assertAlmostEqual(cfg.latent_lam_h, 0.02)
            self.assertAlmostEqual(cfg.latent_lam_p, 0.0)
            self.assertAlmostEqual(cfg.latent_strategy_ppo_coef, 0.30)
            self.assertAlmostEqual(cfg.latent_strategy_aux_return_coef, 1.2)
            self.assertAlmostEqual(cfg.latent_strategy_tau, 0.7)
            self.assertAlmostEqual(cfg.env_win_team_reward, 1.5)
            self.assertAlmostEqual(cfg.env_lose_team_punish, -1.2)
            self.assertAlmostEqual(cfg.env_draw_team_penalty, -0.7)
            self.assertAlmostEqual(cfg.env_action_failed_punishment, -0.02)
            self.assertAlmostEqual(cfg.env_dense_weight, 0.08)
            self.assertEqual(cfg.env_stalemate_max_steps, 120)

    def test_plan_option_a_matches_latent_a1_plan_faithful(self) -> None:
        a = _apply_training_preset(PPOConfig(), "plan_option_a")
        b = _apply_training_preset(PPOConfig(), "latent_a1_plan_faithful")
        self.assertEqual(a.run_tag, b.run_tag)
        self.assertEqual(a.latent_resample_every_n, b.latent_resample_every_n)
        self.assertAlmostEqual(a.latent_lam_p, b.latent_lam_p)

    def test_plan_option_b_lamp_sparse_refresh_and_lam_p(self) -> None:
        cfg = _apply_training_preset(PPOConfig(), "plan_option_b_lamp")
        self.assertEqual(cfg.latent_resample_every_n, 20)
        self.assertAlmostEqual(cfg.latent_lam_p, 0.02)
        self.assertEqual(cfg.run_tag, "plan_option_b_lamp_1m_2v2")
        self.assertTrue(cfg.use_latent_strategy)

    def test_hypothesis_presets_enable_opponent_randomize(self) -> None:
        flat_c = _apply_training_preset(PPOConfig(), "hypothesis_flat_opprand")
        self.assertEqual(flat_c.mode, TrainMode.OPPONENT_POOL.value)
        self.assertTrue(flat_c.opponent_randomize)
        self.assertFalse(flat_c.use_latent_strategy)
        self.assertEqual(flat_c.opponent_pool, ("OP1", "OP2", "OP3"))
        lat_a = _apply_training_preset(PPOConfig(), "hypothesis_latent_opprand_optiona")
        self.assertEqual(lat_a.mode, TrainMode.OPPONENT_POOL.value)
        self.assertTrue(lat_a.opponent_randomize)
        self.assertTrue(lat_a.use_latent_strategy)
        self.assertEqual(lat_a.opponent_pool, ("OP1", "OP2", "OP3"))
        lat_b = _apply_training_preset(PPOConfig(), "hypothesis_latent_opprand_optionb_lamp_coef05")
        self.assertEqual(lat_b.mode, TrainMode.OPPONENT_POOL.value)
        self.assertTrue(lat_b.opponent_randomize)
        self.assertEqual(lat_b.opponent_pool, ("OP1", "OP2", "OP3"))
        self.assertEqual(lat_b.latent_resample_every_n, 20)
        self.assertAlmostEqual(lat_b.latent_strategy_ppo_coef, 0.5)
        no_lamp = _apply_training_preset(PPOConfig(), "hypothesis_latent_opprand_optionb_no_lamp")
        self.assertAlmostEqual(no_lamp.latent_lam_p, 0.0)
        self.assertAlmostEqual(no_lamp.latent_strategy_ppo_coef, 0.5)
        c03 = _apply_training_preset(PPOConfig(), "hypothesis_latent_opprand_optionb_coef03")
        self.assertAlmostEqual(c03.latent_strategy_ppo_coef, 0.3)

    def test_opponent_pool_strips_op4_for_training_by_default(self) -> None:
        cfg = PPOConfig()
        cfg.opponent_pool = ("OP1", "OP2", "OP3", "OP4")
        cfg.allow_op4_in_training_pool = False
        _strip_eval_only_opponents_from_training_pool(cfg)
        self.assertEqual(cfg.opponent_pool, ("OP1", "OP2", "OP3"))

    def test_opponent_pool_keeps_op4_when_allowed(self) -> None:
        cfg = PPOConfig()
        cfg.opponent_pool = ("OP1", "OP4")
        cfg.allow_op4_in_training_pool = True
        _strip_eval_only_opponents_from_training_pool(cfg)
        self.assertEqual(cfg.opponent_pool, ("OP1", "OP4"))

    def test_opponent_randomize_rejected_with_curriculum(self) -> None:
        _WORKSPACE_TMP.mkdir(parents=True, exist_ok=True)
        tag = "unittest_opp_rand_curriculum_conflict_2v2"
        cfg = _smoke_ppo_config(run_tag=tag, checkpoint_dir=str(_WORKSPACE_TMP))
        cfg.mode = "CURRICULUM"
        cfg.use_latent_strategy = False
        cfg.opponent_randomize = True
        try:
            with self.assertRaisesRegex(ValueError, "opponent_randomize=True is incompatible"):
                train_ppo(cfg)
        finally:
            _cleanup_training_outputs(tag)

    def test_plan_faithful_a1_preset_applies_expected_knobs(self) -> None:
        for preset in ("latent_a1_plan_faithful", "latent_op3_a1_plan_faithful"):
            cfg = _apply_training_preset(PPOConfig(), preset)
            self.assertEqual(cfg.total_timesteps, 1_000_000, msg=preset)
            self.assertEqual(cfg.run_tag, "latent_a1_plan_faithful_1m_2v2", msg=preset)
            self.assertIsNone(cfg.load_path, msg=preset)
            self.assertFalse(cfg.latent_strategy_aux_return_head)
            self.assertEqual(cfg.latent_entropy_objective, "maximize")
            self.assertAlmostEqual(cfg.latent_lam_h, 0.005)
            self.assertAlmostEqual(cfg.latent_lam_p, 0.02)
            self.assertAlmostEqual(cfg.latent_strategy_aux_return_coef, 0.0)
            self.assertAlmostEqual(cfg.latent_strategy_tau, 1.0)
            self.assertEqual(cfg.latent_resample_every_n, 0)
            self.assertEqual(cfg.latent_vf_hidden, 128)
            self.assertEqual(cfg.reward_shaping_coef_end, 1.0)
            self.assertEqual(cfg.reward_shaping_decay_steps, 0)
            self.assertIsNone(cfg.env_win_team_reward, msg=preset)
            self.assertIsNone(cfg.env_dense_weight, msg=preset)

    def test_plan_faithful_ablation_presets_apply_expected_knobs(self) -> None:
        main = _apply_training_preset(PPOConfig(), "plan_faithful_latent_persist_entropy")
        self.assertTrue(main.use_latent_strategy)
        self.assertEqual(main.latent_k, 4)
        self.assertEqual(main.latent_resample_every_n, 20)
        self.assertAlmostEqual(main.latent_lam_p, 0.025)
        self.assertAlmostEqual(main.latent_lam_h, 0.003)
        self.assertFalse(main.latent_strategy_aux_return_head)
        self.assertAlmostEqual(main.latent_strategy_aux_return_coef, 0.0)
        self.assertEqual(main.latent_entropy_objective, "maximize")

        no_p = _apply_training_preset(PPOConfig(), "plan_faithful_latent_no_persistence")
        self.assertEqual(no_p.latent_resample_every_n, 20)
        self.assertAlmostEqual(no_p.latent_lam_p, 0.0)

        no_h = _apply_training_preset(PPOConfig(), "plan_faithful_latent_no_entropy")
        self.assertEqual(no_h.latent_entropy_objective, "none")
        self.assertAlmostEqual(no_h.latent_lam_h, 0.0)

        k1 = _apply_training_preset(PPOConfig(), "plan_faithful_latent_k1")
        self.assertEqual(k1.latent_k, 1)

        plain = _apply_training_preset(PPOConfig(), "plan_faithful_no_latent")
        self.assertFalse(plain.use_latent_strategy)
        self.assertFalse(plain.fixed_latent_strategy)

    def test_plan_faithful_latent_episode_z_clean_preset(self) -> None:
        cfg = _apply_training_preset(PPOConfig(), "plan_faithful_latent_episode_z_clean")

        self.assertTrue(cfg.use_latent_strategy)
        self.assertEqual(cfg.latent_k, 4)
        self.assertEqual(cfg.latent_resample_every_n, 0)
        self.assertAlmostEqual(cfg.latent_lam_p, 0.0)
        self.assertAlmostEqual(cfg.latent_lam_h, 0.001)

        self.assertEqual(cfg.latent_strategy_aux_predict_phase_coef, 0.0)
        self.assertFalse(cfg.latent_strategy_aux_return_head)
        self.assertAlmostEqual(cfg.latent_strategy_aux_return_coef, 0.0)

        self.assertEqual(
            cfg.run_tag,
            "plan_faithful_latent_episode_z_clean_1m_2v2"
        )

    def test_plan_faithful_latent_option_a_preset(self) -> None:
        """Fix D: ``plan_faithful_latent_option_a`` must match the Summer-plan Option A recipe exactly."""
        for alias in (
            "plan_faithful_latent_option_a",
            "latent_option_a",
            "plan_faithful_latent_fix_d",
            "latent_fix_d",
        ):
            cfg = _apply_training_preset(PPOConfig(), alias)

            self.assertTrue(cfg.use_latent_strategy, msg=alias)
            self.assertEqual(cfg.latent_k, 4, msg=alias)
            self.assertEqual(cfg.latent_resample_every_n, 0, msg=alias)
            self.assertAlmostEqual(cfg.latent_lam_p, 0.0, msg=alias)
            self.assertAlmostEqual(cfg.latent_lam_h, 0.001, msg=alias)
            self.assertEqual(cfg.latent_entropy_objective, "maximize", msg=alias)
            self.assertAlmostEqual(cfg.latent_strategy_aux_predict_phase_coef, 0.0, msg=alias)
            self.assertFalse(cfg.latent_strategy_aux_return_head, msg=alias)
            self.assertAlmostEqual(cfg.latent_strategy_aux_return_coef, 0.0, msg=alias)
            self.assertFalse(cfg.latent_resample_on_flag, msg=alias)
            self.assertAlmostEqual(cfg.latent_kl_consecutive, 0.0, msg=alias)
            self.assertFalse(cfg.fixed_latent_strategy, msg=alias)
            self.assertEqual(
                cfg.run_tag,
                "plan_faithful_latent_option_a_1m_2v2",
                msg=alias,
            )

    def test_plan_faithful_latent_option_a_episode_credit_preset(self) -> None:
        cfg = _apply_training_preset(PPOConfig(), "plan_faithful_latent_option_a_episode_credit")

        self.assertTrue(cfg.use_latent_strategy)
        self.assertEqual(cfg.latent_k, 4)
        self.assertEqual(cfg.latent_resample_every_n, 0)
        self.assertAlmostEqual(cfg.latent_lam_p, 0.0)
        self.assertAlmostEqual(cfg.latent_lam_h, 0.001)
        self.assertEqual(cfg.latent_entropy_objective, "maximize")
        self.assertEqual(cfg.latent_actor_conditioning, "concat")
        self.assertAlmostEqual(cfg.latent_strategy_aux_predict_phase_coef, 0.0)
        self.assertFalse(cfg.latent_strategy_aux_return_head)
        self.assertAlmostEqual(cfg.latent_strategy_aux_return_coef, 0.0)
        self.assertAlmostEqual(cfg.latent_strategy_ppo_coef, 0.0)
        self.assertTrue(cfg.latent_episode_strategy_ppo)
        self.assertAlmostEqual(cfg.latent_episode_strategy_coef, 0.25)
        self.assertAlmostEqual(cfg.latent_episode_strategy_clip_eps, 0.2)
        self.assertAlmostEqual(cfg.latent_episode_strategy_value_coef, 0.5)
        self.assertTrue(cfg.latent_episode_strategy_return_norm)
        self.assertEqual(cfg.run_tag, "plan_faithful_latent_option_a_episode_credit_1m_2v2")

    def test_plan_faithful_latent_option_a_is_single_flag_contrast_vs_no_latent(self) -> None:
        """Only the latent path (and its hyperparameters) may differ vs ``plan_faithful_no_latent``."""
        latent = _apply_training_preset(PPOConfig(), "plan_faithful_latent_option_a")
        plain = _apply_training_preset(PPOConfig(), "plan_faithful_no_latent")

        latent_only_fields = {
            "use_latent_strategy",
            "latent_k",
            "latent_z_embed_dim",
            "latent_vf_hidden",
            "latent_strategy_hidden",
            "latent_entropy_objective",
            "latent_lam_h",
            "latent_lam_p",
            "latent_strategy_ppo_coef",
            "latent_strategy_aux_return_head",
            "latent_strategy_aux_return_coef",
            "latent_strategy_aux_predict_phase_coef",
            "latent_strategy_tau",
            "latent_resample_every_n",
            "latent_gae_reset_on_z_change",
            "latent_bootstrap_z_deterministic",
            "fixed_latent_strategy",
            "fixed_latent_strategy_id",
            "latent_resample_on_flag",
            "latent_kl_consecutive",
            "run_tag",
        }
        for field_name in latent.__dataclass_fields__:
            if field_name in latent_only_fields:
                continue
            self.assertEqual(
                getattr(latent, field_name),
                getattr(plain, field_name),
                msg=f"plan_faithful_latent_option_a must match plan_faithful_no_latent on non-latent field {field_name!r}",
            )

    def test_plan_faithful_presets_disable_phase_predictor(self) -> None:
        faithful_presets = [
            "plan_faithful_latent_phase1_coupling",
            "plan_faithful_latent_phase2_credit",
            "plan_faithful_latent_phase3b_outcome_clean",
            "plan_faithful_latent_phase4a_rescue",
            "plan_faithful_latent_phase4a_rescue_hardpool",
            "plan_faithful_latent_episode_z_clean",
            "plan_faithful_latent_option_a",
            "plan_faithful_latent_option_a_episode_credit",
        ]

        for preset in faithful_presets:
            cfg = _apply_training_preset(PPOConfig(), preset)
            self.assertEqual(
                cfg.latent_strategy_aux_predict_phase_coef,
                0.0,
                msg=f"{preset} should not use phase prediction"
            )

    def test_wrmax_train_2m_preset(self) -> None:
        cfg = _apply_training_preset(PPOConfig(), "latent_op3_wrmax_train_2m")
        self.assertEqual(cfg.total_timesteps, 2_000_000)
        self.assertEqual(cfg.run_tag, "latent_op3_wrmax_train_2m_2v2")

    def test_gpu_env_reward_kwargs_skips_unset_fields(self) -> None:
        cfg = PPOConfig()
        self.assertEqual(_gpu_env_reward_kwargs(cfg), {})
        cfg.env_win_team_reward = 1.7
        self.assertEqual(_gpu_env_reward_kwargs(cfg), {"win_team_reward": 1.7})
        cfg.env_action_failed_punishment = -0.02
        self.assertEqual(
            _gpu_env_reward_kwargs(cfg),
            {"win_team_reward": 1.7, "action_failed_punishment": -0.02},
        )

    def test_csv_writer_rejects_existing_schema_mismatch(self) -> None:
        _WORKSPACE_TMP.mkdir(parents=True, exist_ok=True)
        path = _WORKSPACE_TMP / "schema_mismatch_episodes.csv"
        try:
            path.write_text("episode_id,phase_name,opponent\n1,OP3,SCRIPTED:OP3\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "CSV schema mismatch"):
                _write_csv_row(
                    str(path),
                    ["episode_id", "opponent"],
                    {"episode_id": 2, "opponent": "SCRIPTED:OP3"},
                )
        finally:
            if path.exists():
                path.unlink()

    def test_csv_writer_migrates_additive_columns(self) -> None:
        """Older metrics CSVs missing newly added telemetry columns rewrite in place then append."""
        _WORKSPACE_TMP.mkdir(parents=True, exist_ok=True)
        path = _WORKSPACE_TMP / "schema_additive_metrics.csv"
        try:
            path.write_text("a,b\n1,2\n", encoding="utf-8")
            _write_csv_row(
                str(path),
                ["a", "x", "b"],
                {"a": "3", "x": "99", "b": "4"},
            )
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0], {"a": "1", "x": "", "b": "2"})
            self.assertEqual(rows[1], {"a": "3", "x": "99", "b": "4"})
        finally:
            if path.exists():
                path.unlink()

    def test_csv_writer_migrates_renamed_strategy_aux_return_loss_column(self) -> None:
        """Metrics CSVs using legacy ``strategy_q_loss`` migrate to ``strategy_aux_return_loss``."""
        _WORKSPACE_TMP.mkdir(parents=True, exist_ok=True)
        path = _WORKSPACE_TMP / "schema_rename_metrics_aux_return.csv"
        try:
            path.write_text("a,strategy_q_loss,b\n1,0.5,2\n", encoding="utf-8")
            _write_csv_row(
                str(path),
                ["a", "strategy_aux_return_loss", "b"],
                {"a": "3", "strategy_aux_return_loss": "0.25", "b": "4"},
                legacy_column_fill=_METRICS_CSV_LEGACY_COLUMN_FILL,
            )
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0], {"a": "1", "strategy_aux_return_loss": "0.5", "b": "2"})
            self.assertEqual(rows[1], {"a": "3", "strategy_aux_return_loss": "0.25", "b": "4"})
        finally:
            if path.exists():
                path.unlink()

    def test_run_lock_blocks_duplicate_run_tag(self) -> None:
        _WORKSPACE_TMP.mkdir(parents=True, exist_ok=True)
        cfg = PPOConfig()
        cfg.checkpoint_dir = str(_WORKSPACE_TMP)
        cfg.run_tag = "unittest_lock_2v2"
        lock = _acquire_run_lock(cfg)
        try:
            with self.assertRaisesRegex(RuntimeError, "Active PPO run lock"):
                _acquire_run_lock(cfg)
        finally:
            lock.release()
        self.assertFalse((_WORKSPACE_TMP / f"{cfg.run_tag}.run.lock").exists())

    def test_train_ppo_smoke_custom_few_steps(self) -> None:
        _run_smoke_and_cleanup(tag="unittest_smoke_custom_ppo_2v2")

    def test_train_ppo_curriculum_starts_at_op1(self) -> None:
        _WORKSPACE_TMP.mkdir(parents=True, exist_ok=True)
        tag = "unittest_curriculum_2v2"
        cfg = _smoke_ppo_config(run_tag=tag, checkpoint_dir=str(_WORKSPACE_TMP))
        cfg.mode = "CURRICULUM"
        cfg.use_latent_strategy = False
        cfg.n_steps = 4
        cfg.total_timesteps = 4
        cfg.max_decision_steps = 1
        final_zip = _WORKSPACE_TMP / f"final_{tag}.zip"
        episode_csv = _WORKSPACE_TMP / f"{tag}_episodes.csv"
        try:
            train_ppo(cfg)
            self.assertTrue(final_zip.is_file())
            with episode_csv.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertGreaterEqual(len(rows), 1)
            self.assertEqual(rows[0]["opponent"], "SCRIPTED:OP1")
            self.assertEqual(rows[0].get("opponent_id", ""), "0")
            self.assertEqual(rows[0]["curriculum_phase"], "OP1")
        finally:
            _cleanup_training_outputs(tag)

    def test_train_ppo_smoke_latent_strategy(self) -> None:
        _WORKSPACE_TMP.mkdir(parents=True, exist_ok=True)
        tag = "unittest_smoke_latent_ppo_2v2"
        cfg = _smoke_ppo_config(run_tag=tag, checkpoint_dir=str(_WORKSPACE_TMP))
        cfg.use_latent_strategy = True
        cfg.latent_resample_every_n = 0
        final_zip = _WORKSPACE_TMP / f"final_{tag}.zip"
        try:
            train_ppo(cfg)
            assert final_zip.is_file(), f"expected {final_zip}"
            payload = _load_checkpoint_payload(final_zip)
            stats = payload.get("last_stats", {})
            self.assertIn("strategy_switch_fraction", stats)
            self.assertIn("strategy_resample_fraction_rollout", stats)
            self.assertIn("strategy_occupancy_0", stats)
            occupancy = [float(stats.get(f"strategy_occupancy_{i}", 0.0)) for i in range(cfg.latent_k)]
            self.assertAlmostEqual(sum(occupancy), 1.0, places=5)
        finally:
            _cleanup_training_outputs(tag)

    def test_default_latent_stays_on_fixed_opponent(self) -> None:
        _WORKSPACE_TMP.mkdir(parents=True, exist_ok=True)
        tag = "unittest_default_latent_fixed_opponent_2v2"
        cfg = _smoke_ppo_config(run_tag=tag, checkpoint_dir=str(_WORKSPACE_TMP))
        cfg.use_latent_strategy = True
        cfg.n_steps = 1
        cfg.total_timesteps = 1
        cfg.max_decision_steps = 1
        episode_csv = _WORKSPACE_TMP / f"{tag}_episodes.csv"
        try:
            train_ppo(cfg)
            with episode_csv.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertGreaterEqual(len(rows), 1)
            self.assertEqual(rows[0]["opponent"], "SCRIPTED:OP3")
            self.assertEqual(rows[0].get("opponent_id", ""), "2")
        finally:
            _cleanup_training_outputs(tag)

    def test_saved_checkpoint_loads_for_local_inference(self) -> None:
        _WORKSPACE_TMP.mkdir(parents=True, exist_ok=True)
        tag = "unittest_inference_custom_ppo_2v2"
        cfg = _smoke_ppo_config(run_tag=tag, checkpoint_dir=str(_WORKSPACE_TMP))
        final_zip = _WORKSPACE_TMP / f"final_{tag}.zip"
        env = None
        try:
            train_ppo(cfg)
            meta = read_custom_ppo_metadata(str(final_zip))
            self.assertEqual(meta["format"], CUSTOM_PPO_FORMAT)
            self.assertEqual(meta["actor_arch"], CUSTOM_PPO_ACTOR_ARCH)
            self.assertEqual(meta["actor_cnn_feature_dim"], cfg.actor_cnn_feature_dim)
            self.assertEqual(meta["vec_schema_version"], CUSTOM_PPO_VEC_SCHEMA_VERSION)
            self.assertEqual(meta["n_blue"], 2)
            self.assertFalse(meta["use_latent_strategy"])
            env = GPUCTFVecEnv(GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu", seed=123))
            obs = env.reset()
            policy = load_custom_ppo_policy(str(final_zip), env.observation_space, env.action_space, device="cpu")
            actions, _ = policy.predict(obs, deterministic=True)
            self.assertEqual(actions.shape, (4,))
            self.assertEqual(policy.strategy_info(), {})
        finally:
            if env is not None:
                env.close()
            _cleanup_training_outputs(tag)

    def test_saved_latent_checkpoint_loads_for_local_inference(self) -> None:
        _WORKSPACE_TMP.mkdir(parents=True, exist_ok=True)
        tag = "unittest_inference_latent_ppo_2v2"
        cfg = _smoke_ppo_config(run_tag=tag, checkpoint_dir=str(_WORKSPACE_TMP))
        cfg.use_latent_strategy = True
        cfg.latent_resample_every_n = 0
        final_zip = _WORKSPACE_TMP / f"final_{tag}.zip"
        env = None
        try:
            train_ppo(cfg)
            meta = read_custom_ppo_metadata(str(final_zip))
            self.assertEqual(meta["format"], CUSTOM_PPO_LATENT_FORMAT)
            self.assertEqual(meta["actor_arch"], CUSTOM_PPO_ACTOR_ARCH)
            self.assertEqual(meta["actor_cnn_feature_dim"], cfg.actor_cnn_feature_dim)
            self.assertEqual(meta["vec_schema_version"], CUSTOM_PPO_VEC_SCHEMA_VERSION)
            self.assertTrue(meta["use_latent_strategy"])
            self.assertEqual(meta["latent_k"], cfg.latent_k)
            env = GPUCTFVecEnv(GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu", seed=123))
            obs = env.reset()
            obs["global_state"] = env.state()
            policy = load_custom_ppo_policy(str(final_zip), env.observation_space, env.action_space, device="cpu")
            actions, _ = policy.predict(obs, deterministic=True)
            self.assertEqual(actions.shape, (4,))
            info = policy.strategy_info()
            self.assertIn("strategy", info)
            self.assertIn("strategy_entropy", info)
            self.assertEqual(info["strategy_k"], cfg.latent_k)
        finally:
            if env is not None:
                env.close()
            _cleanup_training_outputs(tag)

    def test_training_writes_update_and_episode_metrics_csvs(self) -> None:
        _WORKSPACE_TMP.mkdir(parents=True, exist_ok=True)
        tag = "unittest_metrics_latent_ppo_2v2"
        cfg = _smoke_ppo_config(run_tag=tag, checkpoint_dir=str(_WORKSPACE_TMP))
        cfg.use_latent_strategy = True
        cfg.n_steps = 4
        cfg.total_timesteps = 8
        cfg.max_decision_steps = 1
        final_zip = _WORKSPACE_TMP / f"final_{tag}.zip"
        metrics_csv = _WORKSPACE_TMP / f"{tag}_metrics.csv"
        episode_csv = _WORKSPACE_TMP / f"{tag}_episodes.csv"
        try:
            train_ppo(cfg)
            self.assertTrue(final_zip.is_file())
            self.assertTrue(metrics_csv.is_file())
            self.assertTrue(episode_csv.is_file())
            with metrics_csv.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertGreaterEqual(len(rows), 1)
            self.assertIn("run_id", rows[0])
            self.assertIn("run_pid", rows[0])
            self.assertNotEqual(rows[0]["run_id"], "")
            self.assertIn("timesteps", rows[0])
            self.assertIn("rollout_win_rate", rows[0])
            self.assertIn("rolling_win_rate_50ep", rows[0])
            self.assertIn("rolling_win_rate_200ep", rows[0])
            self.assertIn("explained_variance", rows[0])
            self.assertIn("reward_offense_mean", rows[0])
            self.assertIn("reward_sparse_mean", rows[0])
            self.assertIn("reward_failure_mean", rows[0])
            self.assertIn("reward_failure_to_outcome_abs", rows[0])
            self.assertIn("strategy_entropy_frac", rows[0])
            self.assertIn("strategy_wr_spread", rows[0])
            self.assertIn("strategy_occupancy_0", rows[0])
            self.assertIn("latent_mi_z_opponent_nats", rows[0])
            self.assertIn("latent_mi_z_phase_nats", rows[0])
            self.assertIn("latent_mi_z_outcome_nats", rows[0])
            self.assertIn("latent_mi_z_flag_state_nats", rows[0])
            self.assertIn("q_phi_phase0_entropy_mean", rows[0])
            self.assertIn("q_phi_phase0_z0_prob_mean", rows[0])
            self.assertIn("latent_behavior_diversity_l2_mean", rows[0])
            self.assertIn("latent_switch_near_capture_frac", rows[0])
            self.assertIn("latent_z0_behavior_team_spread_mean", rows[0])
            self.assertIn("forced_z_macro_jsd_mean", rows[0])
            self.assertIn("forced_z0_macro_get_flag_prob", rows[0])
            self.assertIn("strategy_occupancy_op0_z0", rows[0])
            self.assertIn("episode_opp0_z0_count", rows[0])
            self.assertIn("episode_z_0_red_score_mean", rows[0])
            comparisons = compare_policy_updates(metrics_csv, before_policy_update=0, after_policy_update=1)
            self.assertEqual([item.column for item in comparisons], list(COMPARISON_COLUMNS))
            self.assertIn("rollout_red_score_mean", format_markdown_table(comparisons))
            with episode_csv.open(newline="", encoding="utf-8") as f:
                ep_rows = list(csv.DictReader(f))
            self.assertGreaterEqual(len(ep_rows), 1)
            self.assertIn("episode_id", ep_rows[0])
            self.assertIn("run_id", ep_rows[0])
            self.assertIn("run_pid", ep_rows[0])
            self.assertIn("policy_update", ep_rows[0])
            self.assertIn("rollout_step", ep_rows[0])
            self.assertIn("latent_z", ep_rows[0])
            self.assertIn("opponent", ep_rows[0])
            self.assertIn("opponent_id", ep_rows[0])
            self.assertIn("reward_sparse", ep_rows[0])
            self.assertIn("reward_failure", ep_rows[0])
            self.assertNotIn("phase_name", ep_rows[0])
            self.assertIn("success", ep_rows[0])
        finally:
            _cleanup_training_outputs(tag)

    def test_episode_strategy_credit_smoke_writes_metrics_and_experience_table(self) -> None:
        _WORKSPACE_TMP.mkdir(parents=True, exist_ok=True)
        tag = "unittest_episode_strategy_credit_2v2"
        cfg = _smoke_ppo_config(run_tag=tag, checkpoint_dir=str(_WORKSPACE_TMP))
        cfg.use_latent_strategy = True
        cfg.latent_resample_every_n = 0
        cfg.latent_lam_p = 0.0
        cfg.latent_lam_h = 0.001
        cfg.latent_strategy_ppo_coef = 0.0
        cfg.latent_episode_strategy_ppo = True
        cfg.latent_episode_strategy_coef = 0.25
        cfg.n_steps = 4
        cfg.batch_size = 4
        cfg.total_timesteps = 4
        cfg.max_decision_steps = 1
        final_zip = _WORKSPACE_TMP / f"final_{tag}.zip"
        metrics_csv = _WORKSPACE_TMP / f"{tag}_metrics.csv"
        strategy_csv = _WORKSPACE_TMP / f"{tag}_strategy_experience.csv"
        try:
            train_ppo(cfg)
            self.assertTrue(final_zip.is_file())
            self.assertTrue(metrics_csv.is_file())
            self.assertTrue(strategy_csv.is_file())
            with metrics_csv.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertGreaterEqual(len(rows), 1)
            row = rows[0]
            self.assertIn("latent_episode_pg_loss", row)
            self.assertIn("latent_episode_v_loss", row)
            self.assertIn("latent_episode_entropy", row)
            self.assertIn("latent_episode_ratio_mean", row)
            self.assertGreater(float(row["latent_episode_count"]), 0.0)
            with strategy_csv.open(newline="", encoding="utf-8") as f:
                strategy_rows = list(csv.DictReader(f))
            self.assertGreaterEqual(len(strategy_rows), 1)
            for col in ("bucket_id", "z", "count", "mean_return", "win_rate", "q_phi_prob_mean", "chosen_freq"):
                self.assertIn(col, strategy_rows[0])
        finally:
            _cleanup_training_outputs(tag)

    def test_latent_strategy_persists_across_rollout_boundaries(self) -> None:
        cfg = _smoke_ppo_config(
            run_tag="unittest_strategy_persistence_2v2",
            checkpoint_dir=str(_WORKSPACE_TMP),
        )
        cfg.use_latent_strategy = True
        cfg.n_steps = 1
        cfg.batch_size = 1
        cfg.max_decision_steps = 100
        env = GPUCTFVecEnv(GPUFieldConfig(n_envs=1, n_agents_per_team=2, max_decision_steps=100, device="cpu", seed=321))
        try:
            trainer = CustomPPOTrainer(
                env,
                cfg,
                learning_rate=1e-4,
                clip_range=0.2,
                ent_coef=0.0,
                n_epochs=1,
                batch_size=1,
                value_clip_range=0.2,
            )
            first = trainer.collect_rollout()
            second = trainer.collect_rollout()
            self.assertTrue(bool(first.fields["z_resampled"][0, 0].item()))
            self.assertFalse(bool(second.fields["z_resampled"][0, 0].item()))
            self.assertEqual(
                int(first.fields["z"][0, 0].item()),
                int(second.fields["z"][0, 0].item()),
            )
        finally:
            env.close()

    def test_fixed_latent_strategy_clamps_z_without_sampling(self) -> None:
        cfg = _smoke_ppo_config(
            run_tag="unittest_fixed_latent_2v2",
            checkpoint_dir=str(_WORKSPACE_TMP),
        )
        cfg.use_latent_strategy = True
        cfg.fixed_latent_strategy = True
        cfg.fixed_latent_strategy_id = 2
        cfg.n_steps = 4
        cfg.batch_size = 4
        cfg.max_decision_steps = 100
        sample_calls: list[int] = []
        original = SharedActorCentralizedCritic.sample_strategy

        def _track(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            sample_calls.append(1)
            return original(self, *args, **kwargs)

        SharedActorCentralizedCritic.sample_strategy = _track  # type: ignore[assignment]
        env = GPUCTFVecEnv(GPUFieldConfig(n_envs=1, n_agents_per_team=2, max_decision_steps=100, device="cpu", seed=654))
        try:
            trainer = CustomPPOTrainer(
                env,
                cfg,
                learning_rate=1e-4,
                clip_range=0.2,
                ent_coef=0.0,
                n_epochs=1,
                batch_size=4,
                value_clip_range=0.2,
            )
            rollout = trainer.collect_rollout()
            self.assertTrue(torch.all(rollout.fields["z"][: rollout.pos] == 2).item())
            self.assertFalse(bool(rollout.fields["z_resampled"][: rollout.pos].any().item()))
            self.assertFalse(bool(rollout.fields["z_persist_mask"][: rollout.pos].any().item()))
            stats = trainer.update(rollout, total_timesteps=4)
            self.assertEqual(stats["strategy_entropy"], 0.0)
            self.assertEqual(stats["strategy_persist_loss"], 0.0)
            self.assertEqual(sample_calls, [])
        finally:
            SharedActorCentralizedCritic.sample_strategy = original  # type: ignore[assignment]
            env.close()

    def test_checkpoint_preserves_strategy_return_normalizer(self) -> None:
        _WORKSPACE_TMP.mkdir(parents=True, exist_ok=True)
        cfg = _smoke_ppo_config(
            run_tag="unittest_strategy_return_norm_2v2",
            checkpoint_dir=str(_WORKSPACE_TMP),
        )
        cfg.use_latent_strategy = True
        cfg.latent_strategy_aux_return_head = True
        path = _WORKSPACE_TMP / "strategy_return_norm.zip"
        env = GPUCTFVecEnv(GPUFieldConfig(n_envs=1, n_agents_per_team=2, max_decision_steps=100, device="cpu", seed=655))
        env2 = GPUCTFVecEnv(GPUFieldConfig(n_envs=1, n_agents_per_team=2, max_decision_steps=100, device="cpu", seed=656))
        try:
            trainer = CustomPPOTrainer(
                env,
                cfg,
                learning_rate=1e-4,
                clip_range=0.2,
                ent_coef=0.0,
                n_epochs=1,
                batch_size=1,
                value_clip_range=0.2,
            )
            trainer._strategy_return_mean = 1.25
            trainer._strategy_return_var = 0.5
            trainer._strategy_return_count = 123.0
            trainer.save(str(path))

            restored = CustomPPOTrainer(
                env2,
                cfg,
                learning_rate=1e-4,
                clip_range=0.2,
                ent_coef=0.0,
                n_epochs=1,
                batch_size=1,
                value_clip_range=0.2,
            )
            restored.load(str(path))

            self.assertAlmostEqual(restored._strategy_return_mean, 1.25)
            self.assertAlmostEqual(restored._strategy_return_var, 0.5)
            self.assertAlmostEqual(restored._strategy_return_count, 123.0)
        finally:
            if path.exists():
                path.unlink()
            env.close()
            env2.close()


if __name__ == "__main__":
    unittest.main()
