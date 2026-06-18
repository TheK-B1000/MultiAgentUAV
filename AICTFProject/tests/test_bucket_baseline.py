"""Regression tests for v3d: context-bucketed q_phi advantage baseline.

The motivating diagnosis (post-v3c): the marginal-over-V baseline
``mean_k V(s, z_k)`` depends on V being well-calibrated for off-policy z
slots, but each z slot only sees value-loss updates for episodes where it was
actually picked (~25% at uniform). So the marginal baseline subtracts noise.
v3d replaces it with an *empirical* per-bucket mean of episode returns
(``mean(R | bucket(s))``) -- standard stratified-sampling variance reduction.

The bucket key is reward-derived (opponent id, flag/score/spread composite,
or their cross product) and shapes the gradient estimator's variance, NEVER
the policy input. q_phi still sees only ``s`` and learns ``pi(z|s)``.

These tests pin:

1. ``BucketBaseline`` math: per-rollout bucket means, EMA across rollouts,
   min-count fallback to global, telemetry fields.
2. ``resolve_bucket_ids`` mode dispatch.
3. v3d preset wiring (bucket_baseline="opponent", ema=0.9, min_count=8,
   inherits v3c router LR / n_epochs / marginal-baseline path).
4. Config defaults (bucket_baseline None, ema 0.9, min_count 8) for
   back-compat.
5. TrainerHyperparams plumbing.
6. Runtime wiring in ``apply_episode_strategy_ppo``: bucket baseline takes
   priority when configured; outside-the-inner-loop call placement (so EMA
   updates happen exactly once per rollout, not once per inner epoch).
"""
from __future__ import annotations

import pathlib
import re
import unittest

import torch

from rl.custom_ppo.latent_bucket_baseline import BucketBaseline, resolve_bucket_ids


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_LATENT_STATE_SRC = (
    _REPO_ROOT / "rl" / "custom_ppo" / "latent_strategy_state.py"
).read_text(encoding="utf-8")
_EPISODE_CREDIT_SRC = (
    _REPO_ROOT / "rl" / "custom_ppo" / "latent" / "credit" / "episode_credit.py"
).read_text(encoding="utf-8")


class BucketBaselineMathTests(unittest.TestCase):
    """Pin the per-bucket / per-rollout / EMA / fallback math."""

    def test_first_rollout_primes_with_rollout_means(self):
        # 3 buckets, 2 episodes each. No prior state, so EMA prime = rollout means.
        # bucket 0: returns [1, 3] -> mean 2
        # bucket 1: returns [4, 6] -> mean 5
        # bucket 2: returns [10, 12] -> mean 11
        baseline = BucketBaseline(ema=0.9, min_count=1)
        returns = torch.tensor([1.0, 3.0, 4.0, 6.0, 10.0, 12.0])
        buckets = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
        result = baseline.update_and_compute(returns, buckets)
        # Each episode gets its own bucket's mean.
        expected = torch.tensor([2.0, 2.0, 5.0, 5.0, 11.0, 11.0])
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_ema_blends_across_rollouts(self):
        # ema=0.5: each rollout contributes 50%. Rollout 1: bucket 0 mean = 2.
        # Rollout 2: bucket 0 returns [10, 14] mean 12 -> new EMA = 0.5*2 + 0.5*12 = 7.
        baseline = BucketBaseline(ema=0.5, min_count=1)
        # Rollout 1
        baseline.update_and_compute(
            torch.tensor([1.0, 3.0]), torch.tensor([0, 0], dtype=torch.long)
        )
        # Rollout 2
        result = baseline.update_and_compute(
            torch.tensor([10.0, 14.0]), torch.tensor([0, 0], dtype=torch.long)
        )
        # Per-episode baseline = EMA-updated bucket 0 mean = 7 for both episodes.
        self.assertTrue(torch.allclose(result, torch.tensor([7.0, 7.0]), atol=1e-6))

    def test_min_count_fallback_uses_global_mean(self):
        # Bucket 99 has only 1 episode (< min_count=3); bucket 0 has 4.
        # Episode 0 (bucket 99) should fall back to global mean.
        # Global rollout mean = (1+3+5+7 + 100) / 5 = 23.2
        baseline = BucketBaseline(ema=1.0, min_count=3)  # ema=1.0 freezes prior; but no prior here
        # Re-prime with ema=0 so the global is purely this rollout's mean,
        # otherwise on the first call the EMA-init branch primes both to the
        # rollout values regardless of ema.
        baseline = BucketBaseline(ema=0.5, min_count=3)
        returns = torch.tensor([100.0, 1.0, 3.0, 5.0, 7.0])
        buckets = torch.tensor([99, 0, 0, 0, 0], dtype=torch.long)
        result = baseline.update_and_compute(returns, buckets)
        global_mean = (100.0 + 1.0 + 3.0 + 5.0 + 7.0) / 5.0
        bucket0_mean = (1.0 + 3.0 + 5.0 + 7.0) / 4.0
        # Episode 0 falls back to global.
        self.assertAlmostEqual(float(result[0].item()), global_mean, places=5)
        # Episodes 1-4 use bucket 0 mean.
        for i in range(1, 5):
            self.assertAlmostEqual(float(result[i].item()), bucket0_mean, places=5)

    def test_unknown_bucket_id_negative_one_is_a_valid_bucket(self):
        # -1 (unknown opponent / pre-v3d records) is just another bucket id.
        # All episodes in bucket -1 should get bucket -1's mean.
        baseline = BucketBaseline(ema=0.9, min_count=1)
        returns = torch.tensor([5.0, 7.0])
        buckets = torch.tensor([-1, -1], dtype=torch.long)
        result = baseline.update_and_compute(returns, buckets)
        self.assertTrue(torch.allclose(result, torch.tensor([6.0, 6.0]), atol=1e-6))

    def test_baseline_is_detached_no_gradient(self):
        # Bucket baselines are empirical reductions of detached returns; they
        # must NEVER be on an autograd graph (would create a duplicate route
        # to the value-head/strategy_encoder gradients).
        baseline = BucketBaseline(ema=0.9, min_count=1)
        returns = torch.tensor([1.0, 3.0], requires_grad=True)
        buckets = torch.tensor([0, 0], dtype=torch.long)
        result = baseline.update_and_compute(returns, buckets)
        self.assertFalse(result.requires_grad)
        self.assertIsNone(result.grad_fn)

    def test_telemetry_fields_populated(self):
        baseline = BucketBaseline(ema=0.5, min_count=2)
        returns = torch.tensor([1.0, 3.0, 100.0])
        buckets = torch.tensor([0, 0, 1], dtype=torch.long)
        baseline.update_and_compute(returns, buckets)
        stats = baseline.last_stats
        self.assertEqual(stats["bucket_count"], 2)
        # Episode in bucket 1 falls back to global -> fallback fraction = 1/3.
        self.assertAlmostEqual(stats["fallback_fraction"], 1.0 / 3.0, places=5)
        self.assertIn(0, stats["per_bucket_count"])
        self.assertIn(1, stats["per_bucket_count"])
        # var_reduction is adv_std / raw_std; should be < 1 because bucket
        # baseline removed real return variance (bucket 0 returns close to each
        # other, bucket 1 is an outlier).
        self.assertLessEqual(stats["variance_reduction_ratio"], 1.0)

    def test_reset_state_clears_emas(self):
        baseline = BucketBaseline(ema=0.5, min_count=1)
        baseline.update_and_compute(
            torch.tensor([10.0, 20.0]), torch.tensor([0, 0], dtype=torch.long)
        )
        baseline.reset_state()
        result = baseline.update_and_compute(
            torch.tensor([1.0, 3.0]), torch.tensor([0, 0], dtype=torch.long)
        )
        # After reset, EMA primes from scratch -> bucket mean = 2, not blended.
        self.assertTrue(torch.allclose(result, torch.tensor([2.0, 2.0]), atol=1e-6))


class BucketBaselineValidationTests(unittest.TestCase):
    def test_rejects_out_of_range_ema(self):
        with self.assertRaises(ValueError):
            BucketBaseline(ema=-0.1, min_count=1)
        with self.assertRaises(ValueError):
            BucketBaseline(ema=1.1, min_count=1)

    def test_min_count_floor_of_one(self):
        # 0 / negative would mean "no fallback ever" which is fine, but the
        # implementation floors to 1 (a singleton-bucket episode always counts).
        baseline = BucketBaseline(ema=0.9, min_count=0)
        self.assertEqual(baseline.min_count, 1)
        baseline = BucketBaseline(ema=0.9, min_count=-5)
        self.assertEqual(baseline.min_count, 1)

    def test_rejects_mismatched_shapes(self):
        baseline = BucketBaseline(ema=0.9, min_count=1)
        with self.assertRaises(ValueError):
            baseline.update_and_compute(
                torch.tensor([1.0, 2.0]),
                torch.tensor([0], dtype=torch.long),
            )


class ResolveBucketIdsTests(unittest.TestCase):
    def test_opponent_mode_returns_opponent_ids(self):
        opp = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
        buc = torch.tensor([10, 20, 30, 40, 50], dtype=torch.long)
        keys = resolve_bucket_ids(mode="opponent", opponent_ids=opp, bucket_ids=buc)
        self.assertTrue(torch.equal(keys, opp))

    def test_bucket_id_mode_returns_bucket_ids(self):
        opp = torch.tensor([0, 1, 2], dtype=torch.long)
        buc = torch.tensor([10, 20, 30], dtype=torch.long)
        keys = resolve_bucket_ids(mode="bucket_id", opponent_ids=opp, bucket_ids=buc)
        self.assertTrue(torch.equal(keys, buc))

    def test_opponent_x_bucket_no_collision_with_bucket_id_space(self):
        # opponent=0,bucket=255 -> key 255. opponent=1,bucket=0 -> key 256.
        # Critical: opponent=0,bucket=255 must NOT collide with opponent=1,bucket=0.
        opp = torch.tensor([0, 1], dtype=torch.long)
        buc = torch.tensor([255, 0], dtype=torch.long)
        keys = resolve_bucket_ids(mode="opponent_x_bucket", opponent_ids=opp, bucket_ids=buc)
        self.assertNotEqual(int(keys[0].item()), int(keys[1].item()))
        self.assertEqual(int(keys[0].item()), 255)
        self.assertEqual(int(keys[1].item()), 256)

    def test_unknown_mode_raises(self):
        opp = torch.zeros(2, dtype=torch.long)
        buc = torch.zeros(2, dtype=torch.long)
        with self.assertRaises(ValueError):
            resolve_bucket_ids(mode="garbage", opponent_ids=opp, bucket_ids=buc)


class V3dConfigDefaultsTests(unittest.TestCase):
    """Back-compat: bare PPOConfig keeps v3c behavior."""

    def test_bucket_baseline_default_is_none(self):
        from rl.train_ppo import PPOConfig

        cfg = PPOConfig()
        self.assertIsNone(cfg.latent_q_phi_bucket_baseline)

    def test_bucket_baseline_ema_default_0_9(self):
        from rl.train_ppo import PPOConfig

        self.assertAlmostEqual(PPOConfig().latent_q_phi_bucket_baseline_ema, 0.9, places=6)

    def test_bucket_baseline_min_count_default_8(self):
        from rl.train_ppo import PPOConfig

        self.assertEqual(PPOConfig().latent_q_phi_bucket_baseline_min_count, 8)


class V3dPresetWiringTests(unittest.TestCase):
    def _v3d_cfg(self):
        from rl.train_ppo import PPOConfig
        from rl.presets import apply_preset

        return apply_preset(PPOConfig(), "plan_faithful_latent_v3d_smart_router")

    def test_v3d_sets_bucket_baseline_opponent(self):
        cfg = self._v3d_cfg()
        self.assertEqual(cfg.latent_q_phi_bucket_baseline, "opponent")

    def test_v3d_sets_ema_0_9(self):
        cfg = self._v3d_cfg()
        self.assertAlmostEqual(float(cfg.latent_q_phi_bucket_baseline_ema), 0.9, places=6)

    def test_v3d_sets_min_count_8(self):
        cfg = self._v3d_cfg()
        self.assertEqual(int(cfg.latent_q_phi_bucket_baseline_min_count), 8)

    def test_v3d_inherits_v3c_router_machinery(self):
        # The v3d change is ONLY the baseline source. All other v3c knobs
        # must be preserved.
        cfg = self._v3d_cfg()
        self.assertEqual(cfg.latent_episode_strategy_n_epochs, 6)
        self.assertIsNotNone(cfg.latent_episode_strategy_lr)
        self.assertAlmostEqual(float(cfg.latent_episode_strategy_lr), 5e-3, places=8)
        self.assertTrue(cfg.latent_q_phi_marginal_baseline)
        self.assertTrue(cfg.latent_episode_strategy_ppo)
        self.assertEqual(cfg.latent_episode_strategy_warmup_decision_steps, 5)
        self.assertEqual(cfg.latent_strategy_ppo_coef, 0.0)
        self.assertEqual(cfg.latent_lam_p, 0.0)
        self.assertEqual(cfg.latent_k, 4)
        self.assertAlmostEqual(float(cfg.latent_lam_h_end), 0.001, places=8)

    def test_v3d_plan_faithful_no_labels_no_aux(self):
        cfg = self._v3d_cfg()
        self.assertEqual(cfg.latent_strategy_aux_predict_phase_coef, 0.0)
        self.assertFalse(cfg.latent_strategy_aux_return_head)
        self.assertFalse(cfg.fixed_latent_strategy)

    def test_v3d_aliases_resolve_to_same_function(self):
        from rl.train_ppo import PPOConfig
        from rl.presets import apply_preset

        cfg_short = apply_preset(PPOConfig(), "latent_v3d")
        cfg_long = apply_preset(PPOConfig(), "plan_faithful_latent_v3d_smart_router")
        self.assertEqual(cfg_short.latent_q_phi_bucket_baseline, cfg_long.latent_q_phi_bucket_baseline)
        self.assertEqual(
            cfg_short.latent_q_phi_bucket_baseline_ema,
            cfg_long.latent_q_phi_bucket_baseline_ema,
        )


class V3dHyperparamsPlumbingTests(unittest.TestCase):
    """All 3 knobs land on TrainerHyperparams unchanged."""

    def _src(self):
        return (
            _REPO_ROOT / "rl" / "custom_ppo" / "trainer_config.py"
        ).read_text(encoding="utf-8")

    def test_dataclass_declares_all_three_fields(self):
        src = self._src()
        self.assertRegex(
            src, r"latent_q_phi_bucket_baseline\s*:\s*Optional\[\s*str\s*\]"
        )
        self.assertRegex(src, r"latent_q_phi_bucket_baseline_ema\s*:\s*float")
        self.assertRegex(src, r"latent_q_phi_bucket_baseline_min_count\s*:\s*int")

    def test_from_ppo_config_plumbs_str_none_passthrough(self):
        src = self._src()
        self.assertRegex(
            src,
            r"latent_q_phi_bucket_baseline\s*=\s*\(\s*\n?\s*"
            r"str\(getattr\(cfg,\s*['\"]latent_q_phi_bucket_baseline['\"],\s*None\)\)"
            r"\s*\n?\s*if\s+getattr\(cfg,\s*['\"]latent_q_phi_bucket_baseline['\"],\s*None\)"
            r"\s*\n?\s*else\s+None",
            msg="from_ppo_config must coerce the bucket-baseline mode string and pass through None.",
        )

    def test_from_ppo_config_min_count_floor_of_one(self):
        src = self._src()
        self.assertRegex(
            src,
            r"latent_q_phi_bucket_baseline_min_count\s*=\s*max\(\s*\n?\s*1\s*,",
        )


class V3dRuntimeWiringTests(unittest.TestCase):
    """Pin where and how the bucket baseline plugs into apply_episode_strategy_ppo."""

    def test_bucket_baseline_computed_outside_inner_epoch_loop(self):
        # Critical correctness: ``update_and_compute`` must fire exactly ONCE
        # per rollout (it advances the EMA state). If it's inside the inner
        # epoch loop, N=6 advances the EMA 6x per update -- silently corrupts
        # the smoothing schedule.
        src = _EPISODE_CREDIT_SRC
        compute_pos = src.find("bucket_baseline_helper.update_and_compute")
        for_loop_pos = src.find("for _ in range(n_inner_epochs)")
        self.assertGreater(compute_pos, 0, "BucketBaseline.update_and_compute must be called")
        self.assertGreater(for_loop_pos, 0, "n_inner_epochs loop must exist")
        self.assertLess(
            compute_pos,
            for_loop_pos,
            msg=(
                "BucketBaseline.update_and_compute must be called BEFORE the "
                "inner-epoch loop so it fires once per rollout, not once per epoch."
            ),
        )

    def test_bucket_baseline_takes_priority_over_marginal_baseline(self):
        # Priority: bucket (v3d) > marginal (v3b/v3c) > legacy V(s, z_picked).
        # Implemented as an if/elif chain on bucket_baseline_vector first.
        self.assertRegex(
            _EPISODE_CREDIT_SRC,
            r"if\s+bucket_baseline_vector\s+is\s+not\s+None\s*:\s*\n\s*v_baseline\s*=\s*bucket_baseline_vector\s*\n\s*elif\s+getattr\(\s*trainer\.cfg\s*,\s*['\"]latent_q_phi_marginal_baseline['\"]",
            msg=(
                "v_baseline selection must check bucket_baseline_vector first "
                "(v3d), then marginal (v3b/v3c), then legacy."
            ),
        )

    def test_batch_exposes_opponent_ids_and_bucket_ids(self):
        # episode_strategy_training_batch must surface both ids for use by
        # resolve_bucket_ids in any bucket mode.
        src = _EPISODE_CREDIT_SRC
        self.assertIn("opponent_ids", src)
        self.assertIn("bucket_ids", src)
        self.assertRegex(
            src,
            r"['\"]opponent_ids['\"]\s*:\s*opponent_ids",
            msg="Training batch dict must include opponent_ids key for v3d.",
        )

    def test_opponent_id_captured_in_episode_record(self):
        # record_episode_strategy_outcome must extract opponent_id from info
        # via opponent_id_int_from_info and persist into the episode record.
        self.assertRegex(
            _EPISODE_CREDIT_SRC,
            r"opponent_id\s*=\s*int\(\s*opponent_id_int_from_info\(\s*trainer\.cfg\s*,\s*info\s*\)\s*\)",
            msg=(
                "record_episode_strategy_outcome must extract opponent_id "
                "from info using opponent_id_int_from_info(cfg, info)."
            ),
        )

    def test_trainer_init_builds_bucket_baseline_when_mode_set(self):
        trainer_src = (
            _REPO_ROOT / "rl" / "custom_ppo" / "trainer.py"
        ).read_text(encoding="utf-8")
        self.assertIn("self.latent_bucket_baseline", trainer_src)
        self.assertRegex(
            trainer_src,
            r"if\s+hparams\.latent_q_phi_bucket_baseline\s+is\s+not\s+None\s*:",
            msg="Trainer init must gate BucketBaseline construction on mode being set.",
        )

    def test_per_bucket_advantage_normalization_wiring(self):
        # Verify the per-bucket advantage normalization logic exists in episode_credit.
        src = _EPISODE_CREDIT_SRC
        self.assertIn("episode_bucket_baseline_keys", src)
        self.assertIn("mode=str(bucket_mode)", src)
        self.assertIn("sub_adv = adv[mask]", src)
        self.assertIn("normalized_adv[mask] = (sub_adv - sub_adv.mean()) / (sub_adv.std(unbiased=False) + 1e-8)", src)


if __name__ == "__main__":
    unittest.main()
