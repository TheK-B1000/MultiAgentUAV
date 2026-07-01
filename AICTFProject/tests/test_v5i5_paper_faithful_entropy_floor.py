"""Focused tests for the v5i5 paper-faithful entropy-floor preset.

v5i5 is a single-axis follow-up to v5i4: the only training mechanism that
changes is the entropy-floor of the ``q_phi`` regularizer
(``latent_lam_h_end``), raised from ``0.0002`` to ``0.001``. Everything
else -- inheritance, actor pathway, loss objective, sampling cadence,
forbidden-channel set -- must be bit-for-bit identical to v5i4.

These tests pin the v5i5 contract end-to-end:

1. Inheritance: v5i5 derives from v5i4 (NOT from v5i1/v5i2/v5i3) and
   the resolved diff vs v5i4 is exactly ``{latent_lam_h_end, run_tag}``.
2. Concat-only actor: FiLM / adapter / one-hot all OFF (inherited from
   v5i4); actor input dim still 164.
3. No-curriculum: forced-z resolves to zero at every step.
4. Sparse 64-step resampling and ``latent_resample_on_flag is False``.
5. Forbidden channels OFF: episode-credit, arc-credit, aux heads,
   preferences, distillation, specialist router, behavior contrast.
6. Entropy-floor stronger than v5i4: ``latent_lam_h_end == 0.001``,
   start unchanged at ``0.003``, anneal window unchanged.
7. Aliases: every v5i5 alias resolves to the same dict.
8. Banner: ``_maybe_print_paper_faithful_audit`` emits the v5i5
   invariant block (family-prefixed) and flags v5i5-specific
   mis-configurations under the same warning words.
9. Diagnostic field schema: the new occupancy-collapse columns
   (``effective_num_latents``, ``latent_occupancy_min``,
   ``latent_occupancy_max``, ``latent_occupancy_ratio``,
   ``latent_marginal_entropy_nats``, ``mean_strategy_duration``)
   appear in the metrics CSV header. They are pure functions of the
   per-z counts already computed for v5i4 -- no new gradient channel.

These are all readonly / pure-Python checks; no env, no torch optim
state, no checkpoints. They run in well under a second.
"""

from __future__ import annotations

import dataclasses
import io
import unittest
from contextlib import redirect_stdout

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.csv_writers import _update_fieldnames
from rl.custom_ppo.schedules import resolve_latent_forced_z_frac
from rl.presets import apply_preset
from rl.presets.plan_faithful import (
    apply_plan_faithful_latent_v5i4_end_to_end,
    apply_plan_faithful_latent_v5i5_paper_faithful_entropy_floor,
)
from rl.training.banner import _maybe_print_paper_faithful_audit


V5I5_ALIASES = (
    "v5i5",
    "v5i5_paper_faithful",
    "v5i5_paper_faithful_entropy_floor",
    "v5i5_entropy_floor",
    "paper_faithful_entropy_floor",
    "latent_v5i5_paper_faithful",
    "latent_v5i5_paper_faithful_entropy_floor",
    "latent_v5i5_entropy_floor",
    "plan_faithful_latent_v5i5_paper_faithful_entropy_floor",
    "plan_faithful_latent_v5i5_entropy_floor",
)


def _resolved(name: str) -> PPOConfig:
    return apply_preset(PPOConfig(), name)


class V5i5PresetInheritanceTests(unittest.TestCase):
    """v5i5 must derive from v5i4 with exactly one knob changed."""

    def test_v5i5_minimal_diff_vs_v5i4(self) -> None:
        v5i4 = dataclasses.asdict(apply_plan_faithful_latent_v5i4_end_to_end(PPOConfig()))
        v5i5 = dataclasses.asdict(
            apply_plan_faithful_latent_v5i5_paper_faithful_entropy_floor(PPOConfig())
        )
        diffs = {
            k: (v5i4.get(k), v5i5.get(k))
            for k in set(v5i4) | set(v5i5)
            if v5i4.get(k) != v5i5.get(k)
        }
        self.assertEqual(
            set(diffs),
            {"latent_lam_h_end", "run_tag"},
            f"v5i5 must differ from v5i4 only in (latent_lam_h_end, run_tag); "
            f"unexpected diff: {sorted(diffs)}",
        )
        self.assertAlmostEqual(float(v5i4["latent_lam_h_end"]), 0.0002)
        self.assertAlmostEqual(float(v5i5["latent_lam_h_end"]), 0.001)

    def test_v5i5_run_tag_advertises_entropy_floor_and_correct_budget(self) -> None:
        cfg = _resolved("v5i5")
        self.assertEqual(int(cfg.total_timesteps), 1_000_000)
        self.assertIn("v5i5_paper_faithful_entropy_floor", cfg.run_tag)
        self.assertIn("OP5_OP6_OP7", cfg.run_tag)
        self.assertIn("_1m_", cfg.run_tag)
        self.assertNotIn(
            "_2m_",
            cfg.run_tag,
            "v5i5 must not carry the misleading _2m_ suffix that the "
            "v4i1 / v5_strict_summer chain inherited.",
        )

    def test_v5i5_inherits_v5i4_entropy_anneal_window_and_start(self) -> None:
        cfg = _resolved("v5i5")
        self.assertAlmostEqual(float(cfg.latent_lam_h_start), 0.003)
        self.assertAlmostEqual(float(cfg.latent_lam_h), 0.003)
        self.assertEqual(int(cfg.latent_entropy_anneal_start), 0)
        self.assertEqual(int(cfg.latent_entropy_anneal_end), 300_000)
        self.assertEqual(str(cfg.latent_entropy_objective), "maximize")

    def test_v5i5_keeps_v5i4_persistence_and_router_ppo(self) -> None:
        cfg = _resolved("v5i5")
        self.assertAlmostEqual(float(cfg.latent_lam_p), 0.03)
        self.assertAlmostEqual(float(cfg.latent_strategy_ppo_coef), 0.10)


class V5i5ConcatOnlyActorTests(unittest.TestCase):
    """The v5i5 actor pathway is identical to v5i4: CNN(128) + per-agent(20)
    + z_emb(16) = 164. Nothing about the actor changes for v5i5."""

    def test_actor_input_dim_is_164(self) -> None:
        cfg = _resolved("v5i5")
        cnn = int(cfg.actor_cnn_feature_dim)
        vec = 20
        z_emb = int(cfg.latent_z_embed_dim)
        self.assertEqual(cnn + vec + z_emb, 164)
        self.assertEqual(cnn, 128)
        self.assertEqual(z_emb, 16)

    def test_no_actor_z_film_adapter_or_onehot(self) -> None:
        cfg = _resolved("v5i5")
        self.assertFalse(bool(cfg.enable_actor_z_film))
        self.assertFalse(bool(cfg.latent_actor_z_adapter_enabled))
        self.assertFalse(bool(cfg.latent_actor_z_onehot_enabled))
        self.assertAlmostEqual(float(cfg.actor_z_film_init_scale), 0.0)
        self.assertAlmostEqual(float(cfg.latent_actor_z_adapter_scale), 0.0)
        self.assertAlmostEqual(float(cfg.latent_actor_z_onehot_scale), 0.0)


class V5i5NoCurriculumTests(unittest.TestCase):
    """The resolved forced-z fraction must be exactly zero at every step,
    inherited verbatim from v5i4."""

    def test_forced_z_resolves_to_zero_at_every_step(self) -> None:
        cfg = _resolved("v5i5")
        for step in (0, 10_000, 200_000, 500_000, 1_000_000):
            self.assertAlmostEqual(
                resolve_latent_forced_z_frac(cfg, global_step=step),
                0.0,
                places=8,
                msg=f"forced-z must be 0 at step={step}",
            )


class V5i5SparseResamplingTests(unittest.TestCase):
    """v5i5 resamples every 64 decisions, never on flag events."""

    def test_sparse_64_step_resampling(self) -> None:
        cfg = _resolved("v5i5")
        self.assertEqual(int(cfg.latent_resample_every_n), 64)
        self.assertFalse(bool(cfg.latent_resample_on_flag))
        self.assertFalse(bool(cfg.fixed_latent_strategy))
        self.assertTrue(bool(cfg.use_latent_strategy))
        self.assertEqual(int(cfg.latent_k), 4)
        self.assertTrue(bool(cfg.latent_gae_reset_on_z_change))


class V5i5NoForbiddenChannelsTests(unittest.TestCase):
    """Every non-paper q_phi gradient channel must be OFF in v5i5
    (inherited verbatim from v5i4 -- v5i5 deliberately changes only the
    entropy floor, no new gradient channel)."""

    def test_episode_credit_off(self) -> None:
        cfg = _resolved("v5i5")
        self.assertFalse(bool(cfg.latent_episode_strategy_ppo))
        self.assertAlmostEqual(float(cfg.latent_episode_strategy_coef), 0.0)
        self.assertIsNone(cfg.latent_episode_strategy_lr)

    def test_arc_credit_off(self) -> None:
        cfg = _resolved("v5i5")
        self.assertFalse(bool(cfg.latent_arc_credit_enabled))
        self.assertAlmostEqual(float(cfg.latent_arc_credit_coef), 0.0)

    def test_aux_heads_off(self) -> None:
        cfg = _resolved("v5i5")
        self.assertFalse(bool(cfg.latent_strategy_aux_return_head))
        self.assertAlmostEqual(float(cfg.latent_strategy_aux_return_coef), 0.0)
        self.assertAlmostEqual(
            float(cfg.latent_strategy_aux_predict_phase_coef), 0.0
        )

    def test_preferences_and_distillation_off(self) -> None:
        cfg = _resolved("v5i5")
        self.assertFalse(bool(cfg.latent_v3i3_event_preference_enabled))
        self.assertAlmostEqual(float(cfg.latent_v3i3_event_preference_coef), 0.0)
        self.assertAlmostEqual(float(cfg.latent_preference_coef), 0.0)
        self.assertAlmostEqual(float(cfg.latent_preference_commit_coef), 0.0)
        self.assertFalse(bool(cfg.latent_router_distill_enabled))

    def test_specialist_router_and_separation_off(self) -> None:
        cfg = _resolved("v5i5")
        self.assertFalse(bool(cfg.latent_specialist_router_enabled))
        self.assertAlmostEqual(float(cfg.latent_behavior_contrast_coef), 0.0)
        self.assertAlmostEqual(float(cfg.latent_actor_z_separation_coef), 0.0)
        self.assertAlmostEqual(float(cfg.latent_marginal_balance_coef), 0.0)
        self.assertAlmostEqual(
            float(cfg.latent_conditional_entropy_min_coef), 0.0
        )


class V5i5AliasSnapshotTests(unittest.TestCase):
    """Every v5i5 alias resolves to the same config dict."""

    def test_all_aliases_resolve_to_identical_config(self) -> None:
        baseline = dataclasses.asdict(_resolved(V5I5_ALIASES[0]))
        for name in V5I5_ALIASES[1:]:
            current = dataclasses.asdict(_resolved(name))
            self.assertEqual(
                current,
                baseline,
                f"alias {name!r} must resolve to the same config as "
                f"{V5I5_ALIASES[0]!r}",
            )


class V5i5PaperFaithfulAuditBannerTests(unittest.TestCase):
    """The launch-time audit banner emits the v5i5 invariant block when
    the run is configured per the v5i5 contract, prefixing the family
    label so reviewers can see which preset family was matched."""

    def _capture(self, cfg: PPOConfig) -> str:
        buf = io.StringIO()
        with redirect_stdout(buf):
            _maybe_print_paper_faithful_audit(cfg)
        return buf.getvalue()

    def test_audit_banner_fires_for_v5i5(self) -> None:
        cfg = _resolved("v5i5")
        out = self._capture(cfg)
        self.assertIn("v5i5 paper-faithful audit", out)
        self.assertIn("discrete shared z: K=4", out)
        self.assertIn("actor conditioning: embedding-concat", out)
        self.assertIn("FiLM: OFF", out)
        self.assertIn("q_phi task-reward PPO: ON", out)
        self.assertIn("episode-credit extension: OFF", out)
        self.assertIn("forced-z curriculum: OFF", out)
        self.assertIn("auxiliary heads: OFF", out)
        self.assertIn("preferences/distillation: OFF", out)
        self.assertIn("persistence: ON", out)
        self.assertIn("entropy maximization: ON", out)
        self.assertIn("resampling cadence: every 64 decisions", out)
        # The v5i4 banner header must NOT appear when running v5i5.
        self.assertNotIn("v5i4 paper-faithful audit", out)
        # No warning lines should fire for a correctly-configured v5i5.
        self.assertNotIn("v5i5 audit WARNING", out)
        self.assertNotIn("v5i4 audit WARNING", out)

    def test_audit_banner_warns_on_missing_strategy_ppo_coef(self) -> None:
        cfg = _resolved("v5i5")
        cfg.latent_strategy_ppo_coef = 0.0
        out = self._capture(cfg)
        self.assertIn("v5i5 audit WARNING", out)
        self.assertIn("latent_strategy_ppo_coef <= 0", out)

    def test_audit_banner_warns_on_dedicated_router_optimizer(self) -> None:
        cfg = _resolved("v5i5")
        cfg.latent_episode_strategy_lr = 5e-3
        out = self._capture(cfg)
        self.assertIn("v5i5 audit WARNING", out)
        self.assertIn("latent_episode_strategy_lr is set", out)

    def test_audit_banner_warns_on_film(self) -> None:
        cfg = _resolved("v5i5")
        cfg.enable_actor_z_film = True
        out = self._capture(cfg)
        self.assertIn("v5i5 audit WARNING", out)
        self.assertIn("actor-z pathway is not concat-only", out)


class V5i5OccupancyDiagnosticSchemaTests(unittest.TestCase):
    """The new occupancy-collapse diagnostic columns must appear in the
    metrics CSV header for any latent run, so a v5i4-vs-v5i5 comparison
    has the data to distinguish "stronger entropy preserves useful
    diversity" from "stronger entropy makes the router randomly
    uncertain". These columns are pure functions of existing per-z
    counts -- they add no new gradient channel and no new objective
    term, only telemetry."""

    REQUIRED_COLUMNS = (
        "effective_num_latents",
        "latent_marginal_entropy_nats",
        "latent_occupancy_min",
        "latent_occupancy_max",
        "latent_occupancy_ratio",
        "mean_strategy_duration",
    )

    def test_metrics_csv_header_includes_occupancy_diagnostics(self) -> None:
        fields = _update_fieldnames(use_latent_strategy=True, latent_k=4)
        for column in self.REQUIRED_COLUMNS:
            self.assertIn(
                column,
                fields,
                f"metrics CSV header must include {column!r} for v5i5 "
                "occupancy-collapse diagnostics",
            )

    def test_metrics_csv_header_already_includes_per_z_columns(self) -> None:
        """Existing per-z and per-z-per-opponent CSV columns must remain
        in the metrics header so the v5i5 diagnostics table does not
        regress what v5i3 / v5i4 already emit."""
        fields = _update_fieldnames(use_latent_strategy=True, latent_k=4)
        for k in range(4):
            self.assertIn(f"strategy_occupancy_{k}", fields)
            self.assertIn(f"router_sample_count_by_z_{k}", fields)
            self.assertIn(f"episode_count_by_z_{k}", fields)
        for o in range(7):
            for k in range(4):
                self.assertIn(f"strategy_occupancy_op{o}_z{k}", fields)
                self.assertIn(f"episode_opp{o}_z{k}_count", fields)
        # And the rollout-level switch / unique-count columns.
        for column in (
            "strategy_unique_count",
            "strategy_switch_fraction",
            "strategy_resample_fraction_rollout",
        ):
            self.assertIn(column, fields)


if __name__ == "__main__":
    unittest.main()
