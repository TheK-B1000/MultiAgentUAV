"""Focused tests for the v5i4 paper-faithful conditional-entropy row.

v5i4 is built directly on v5_strict_summer (NOT on v5i1/v5i2/v5i3) with
one correction: the on-policy categorical PPO term on ``q_phi`` is
enabled via ``latent_strategy_ppo_coef = 0.10`` so the router actually
learns from task reward. v5i6 inherits this contract and is the current
canonical Summer interpretation because it changes the entropy reduction
to batch-marginal entropy.

These tests pin the v5i4 contract end-to-end:

1. Preset inheritance: v5i4 derives from strict-Summer, not v5i1/v5i2/v5i3.
2. Concat-only actor: FiLM / adapter / one-hot are all OFF.
3. No-curriculum: the resolved forced-z fraction is zero at every step.
4. Router task-gradient is ON: nonzero advantages produce nonzero
   gradient through the main-loop categorical PPO term.
5. Zero-advantage: with zero advantages, the policy_loss component of the
   strategy PPO term is exactly zero (it is the only task-reward channel
   that should flip on/off with advantages in v5i4).
6. No forbidden channels: episode-credit, arc-credit, aux heads,
   preferences, distillation are all OFF.
7. Sparse resampling: ``latent_resample_every_n == 64`` and
   ``latent_resample_on_flag is False``.
8. Snapshot: every v5i4 alias resolves to the same dict.
9. Banner: ``_maybe_print_paper_faithful_audit`` emits the expected
   invariant block and flags the documented mis-configurations.

These are all readonly / pure-Python checks; no env, no torch optim
state, no checkpoints. They run in well under a second.
"""

from __future__ import annotations

import dataclasses
import io
import unittest
from contextlib import redirect_stdout

import torch

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.schedules import resolve_latent_forced_z_frac
from rl.latent_losses import strategy_ppo_loss
from rl.presets import apply_preset
from rl.presets.plan_faithful import (
    apply_plan_faithful_latent_v5_strict_summer,
    apply_plan_faithful_latent_v5i1_reward_credit_router,
    apply_plan_faithful_latent_v5i2_stronger_z_conditioning,
    apply_plan_faithful_latent_v5i3_balanced_warmup,
    apply_plan_faithful_latent_v5i4_end_to_end,
)
from rl.train_ppo import _resolve_initial_opponent_and_phase
from rl.training.banner import _maybe_print_paper_faithful_audit


DEVICE = torch.device("cpu")

V5I4_ALIASES = (
    "v5i4",
    "v5i4_paper_faithful",
    "v5i4_end_to_end",
    "paper_faithful_end_to_end",
    "latent_v5i4_paper_faithful",
    "latent_v5i4_end_to_end",
    "plan_faithful_latent_v5i4_end_to_end",
)


def _resolved(name: str) -> PPOConfig:
    return apply_preset(PPOConfig(), name)


class V5i4PresetInheritanceTests(unittest.TestCase):
    """v5i4 must branch from strict-Summer, not from v5i1/v5i2/v5i3."""

    def test_strict_summer_to_v5i4_is_strategy_ppo_coef_only_for_the_core_qphi_channel(
        self,
    ) -> None:
        """v5i4 turns the strict-Summer baseline operational by setting
        ``latent_strategy_ppo_coef = 0.10`` -- the per-step categorical
        PPO term that lets q_phi see task reward. Other v5_strict_summer
        defaults (concat actor, no FiLM, no aux, no curriculum) carry
        through unchanged.
        """
        strict = apply_plan_faithful_latent_v5_strict_summer(PPOConfig())
        v5i4 = apply_plan_faithful_latent_v5i4_end_to_end(PPOConfig())

        self.assertAlmostEqual(float(strict.latent_strategy_ppo_coef), 0.0)
        self.assertAlmostEqual(float(v5i4.latent_strategy_ppo_coef), 0.10)

        # The actor-z pathway is identical to v5_strict_summer (plain
        # nn.Embedding(K, d_z) concat -- no FiLM, no adapter, no one-hot).
        for field in (
            "enable_actor_z_film",
            "latent_actor_z_adapter_enabled",
            "latent_actor_z_onehot_enabled",
        ):
            self.assertFalse(
                bool(getattr(v5i4, field)),
                f"v5i4 must keep {field}=False (concat-only actor)",
            )
        for field in (
            "latent_z_embed_dim",
            "latent_actor_z_embed_scale",
        ):
            self.assertEqual(getattr(strict, field), getattr(v5i4, field))

    def test_v5i4_does_not_inherit_v5i1_episode_credit_machinery(self) -> None:
        """v5i1 turns the episode-credit extension on and creates a
        dedicated router AdamW via ``latent_episode_strategy_lr``. v5i4
        must keep both OFF or the main-loop PG gate would be silenced.
        """
        v5i1 = apply_plan_faithful_latent_v5i1_reward_credit_router(PPOConfig())
        v5i4 = apply_plan_faithful_latent_v5i4_end_to_end(PPOConfig())

        self.assertTrue(bool(v5i1.latent_episode_strategy_ppo))
        self.assertIsNotNone(v5i1.latent_episode_strategy_lr)

        self.assertFalse(bool(v5i4.latent_episode_strategy_ppo))
        self.assertAlmostEqual(float(v5i4.latent_episode_strategy_coef), 0.0)
        self.assertIsNone(v5i4.latent_episode_strategy_lr)

    def test_v5i4_does_not_inherit_v5i2_film(self) -> None:
        """v5i2 enables FiLM. v5i4 must keep it OFF."""
        v5i2 = apply_plan_faithful_latent_v5i2_stronger_z_conditioning(PPOConfig())
        v5i4 = apply_plan_faithful_latent_v5i4_end_to_end(PPOConfig())

        self.assertTrue(bool(v5i2.enable_actor_z_film))
        self.assertFalse(bool(v5i4.enable_actor_z_film))
        self.assertAlmostEqual(float(v5i4.actor_z_film_init_scale), 0.0)

    def test_v5i4_does_not_inherit_v5i3_forced_z_anneal(self) -> None:
        """v5i3 sets the 0.30 -> 0.00 forced-z anneal across 200k -> 500k
        steps. v5i4 must clear that schedule and pin the legacy frac to
        zero so the actor sees only on-policy rollouts.
        """
        v5i3 = apply_plan_faithful_latent_v5i3_balanced_warmup(PPOConfig())
        v5i4 = apply_plan_faithful_latent_v5i4_end_to_end(PPOConfig())

        self.assertIsNotNone(v5i3.latent_forced_z_episode_frac_start)
        self.assertAlmostEqual(
            float(v5i3.latent_forced_z_episode_frac_start or 0.0),
            0.30,
            places=6,
        )

        self.assertIsNone(v5i4.latent_forced_z_episode_frac_start)
        self.assertIsNone(v5i4.latent_forced_z_episode_frac_end)
        self.assertIsNone(v5i4.latent_forced_z_anneal_start)
        self.assertIsNone(v5i4.latent_forced_z_anneal_end)
        self.assertAlmostEqual(float(v5i4.latent_forced_z_episode_frac), 0.0)


class V5i4ConcatOnlyActorTests(unittest.TestCase):
    """The v5i4 actor pathway is the literal Summer-plan one:
    ``CNN(128) + per_agent_vec(20) + z_emb(16) = 164``."""

    def test_actor_input_dim_is_164(self) -> None:
        cfg = _resolved("v5i4")
        cnn = int(cfg.actor_cnn_feature_dim)
        vec = 20
        z_emb = int(cfg.latent_z_embed_dim)
        self.assertEqual(cnn + vec + z_emb, 164)
        self.assertEqual(cnn, 128)
        self.assertEqual(z_emb, 16)

    def test_no_actor_z_film_adapter_or_onehot(self) -> None:
        cfg = _resolved("v5i4")
        self.assertFalse(bool(cfg.enable_actor_z_film))
        self.assertFalse(bool(cfg.latent_actor_z_adapter_enabled))
        self.assertFalse(bool(cfg.latent_actor_z_onehot_enabled))
        self.assertAlmostEqual(float(cfg.actor_z_film_init_scale), 0.0)
        self.assertAlmostEqual(float(cfg.latent_actor_z_adapter_scale), 0.0)
        self.assertAlmostEqual(float(cfg.latent_actor_z_onehot_scale), 0.0)


class V5i4NoCurriculumTests(unittest.TestCase):
    """The resolved forced-z fraction must be exactly zero at every step,
    so the schedule resolver never tilts q_phi exposure for v5i4.
    """

    def test_forced_z_resolves_to_zero_at_every_step(self) -> None:
        cfg = _resolved("v5i4")
        for step in (0, 10_000, 200_000, 500_000, 1_000_000, 2_000_000):
            self.assertAlmostEqual(
                resolve_latent_forced_z_frac(cfg, global_step=step),
                0.0,
                places=8,
                msg=f"forced-z must be 0 at step={step}",
            )


class V5i4RouterTaskGradientTests(unittest.TestCase):
    """The v5i4 q_phi PPO term must transmit nonzero gradient when there
    is task-reward advantage to credit -- this is *the* paper-faithful
    learning channel for the router. When advantages are zero, the term
    contributes no policy_loss (only entropy/persistence -- handled by
    other tests -- can still update q_phi)."""

    def _make_random_resample_batch(self) -> dict:
        torch.manual_seed(0)
        n = 32
        resample = torch.zeros(n, dtype=torch.bool)
        resample[::4] = True  # 8 resample steps -- matches sparse cadence
        logp = torch.randn(n, requires_grad=True)
        logp_old = logp.detach().clone()
        return {
            "logp": logp,
            "logp_old": logp_old,
            "resample": resample,
        }

    def test_nonzero_advantage_produces_nonzero_qphi_gradient(self) -> None:
        cfg = _resolved("v5i4")
        coef = float(cfg.latent_strategy_ppo_coef)
        self.assertGreater(
            coef,
            0.0,
            "v5i4 must enable the main-loop categorical PPO term",
        )
        batch = self._make_random_resample_batch()
        advantages = torch.randn(batch["logp"].shape[0])
        # Make sure resample-subset advantages are not all equal so the
        # within-subset normalization does not collapse to zero.
        advantages[batch["resample"]] = torch.tensor(
            [1.5, -0.7, 0.3, -1.1, 0.9, -0.4, 1.2, -0.8],
            dtype=torch.float32,
        )
        loss, stats = strategy_ppo_loss(
            batch["logp"],
            batch["logp_old"],
            advantages,
            batch["resample"],
            clip_range=float(cfg.clip_range),
            coef=coef,
            device=DEVICE,
        )
        loss.backward()
        self.assertIsNotNone(batch["logp"].grad)
        grad_norm = float(batch["logp"].grad.norm())  # type: ignore[union-attr]
        self.assertGreater(
            grad_norm,
            0.0,
            "Nonzero advantages must produce nonzero q_phi gradient under v5i4",
        )
        # And the gradient must concentrate on the resample subset.
        non_resample_grad_sum = float(
            batch["logp"].grad[~batch["resample"]].abs().sum()  # type: ignore[index]
        )
        self.assertAlmostEqual(non_resample_grad_sum, 0.0, places=6)

    def test_zero_advantage_produces_zero_policy_loss(self) -> None:
        """With zero advantages, the categorical PPO term contributes zero
        policy_loss. Entropy/persistence -- which still run independently
        via the main-loop coef-gate -- are tested in the dedicated
        ``MainLoopGatingTests`` and are NOT exercised here."""
        cfg = _resolved("v5i4")
        batch = self._make_random_resample_batch()
        advantages = torch.zeros(batch["logp"].shape[0])
        loss, stats = strategy_ppo_loss(
            batch["logp"],
            batch["logp_old"],
            advantages,
            batch["resample"],
            clip_range=float(cfg.clip_range),
            coef=float(cfg.latent_strategy_ppo_coef),
            device=DEVICE,
        )
        self.assertAlmostEqual(float(loss.item()), 0.0, places=7)
        self.assertAlmostEqual(float(stats["policy_loss"].item()), 0.0, places=7)


class V5i4NoForbiddenChannelsTests(unittest.TestCase):
    """Every non-paper q_phi gradient channel must be OFF in v5i4."""

    def test_episode_credit_off(self) -> None:
        cfg = _resolved("v5i4")
        self.assertFalse(bool(cfg.latent_episode_strategy_ppo))
        self.assertAlmostEqual(float(cfg.latent_episode_strategy_coef), 0.0)
        self.assertIsNone(cfg.latent_episode_strategy_lr)

    def test_arc_credit_off(self) -> None:
        cfg = _resolved("v5i4")
        self.assertFalse(bool(cfg.latent_arc_credit_enabled))
        self.assertAlmostEqual(float(cfg.latent_arc_credit_coef), 0.0)

    def test_aux_heads_off(self) -> None:
        cfg = _resolved("v5i4")
        self.assertFalse(bool(cfg.latent_strategy_aux_return_head))
        self.assertAlmostEqual(float(cfg.latent_strategy_aux_return_coef), 0.0)
        self.assertAlmostEqual(
            float(cfg.latent_strategy_aux_predict_phase_coef), 0.0
        )

    def test_preferences_and_distillation_off(self) -> None:
        cfg = _resolved("v5i4")
        self.assertFalse(bool(cfg.latent_v3i3_event_preference_enabled))
        self.assertAlmostEqual(float(cfg.latent_v3i3_event_preference_coef), 0.0)
        self.assertAlmostEqual(float(cfg.latent_preference_coef), 0.0)
        self.assertAlmostEqual(float(cfg.latent_preference_commit_coef), 0.0)
        self.assertFalse(bool(cfg.latent_router_distill_enabled))

    def test_specialist_router_and_separation_off(self) -> None:
        cfg = _resolved("v5i4")
        self.assertFalse(bool(cfg.latent_specialist_router_enabled))
        self.assertAlmostEqual(float(cfg.latent_behavior_contrast_coef), 0.0)
        self.assertAlmostEqual(float(cfg.latent_actor_z_separation_coef), 0.0)
        self.assertAlmostEqual(float(cfg.latent_marginal_balance_coef), 0.0)
        self.assertAlmostEqual(
            float(cfg.latent_conditional_entropy_min_coef), 0.0
        )

    def test_persistence_and_entropy_on(self) -> None:
        cfg = _resolved("v5i4")
        self.assertGreater(float(cfg.latent_lam_p), 0.0)
        self.assertGreater(float(cfg.latent_lam_h), 0.0)
        self.assertEqual(str(cfg.latent_entropy_objective), "maximize")
        self.assertAlmostEqual(float(cfg.latent_kl_consecutive), 0.0)


class V5i4SparseResamplingTests(unittest.TestCase):
    """v5i4 resamples every 64 decisions, never on flag events."""

    def test_sparse_64_step_resampling(self) -> None:
        cfg = _resolved("v5i4")
        self.assertEqual(int(cfg.latent_resample_every_n), 64)
        self.assertFalse(bool(cfg.latent_resample_on_flag))
        self.assertFalse(bool(cfg.fixed_latent_strategy))
        self.assertTrue(bool(cfg.use_latent_strategy))
        self.assertEqual(int(cfg.latent_k), 4)


class V5i4AliasSnapshotTests(unittest.TestCase):
    """Every v5i4 alias resolves to the same config dict."""

    def test_all_aliases_resolve_to_identical_config(self) -> None:
        baseline = dataclasses.asdict(_resolved(V5I4_ALIASES[0]))
        for name in V5I4_ALIASES[1:]:
            current = dataclasses.asdict(_resolved(name))
            self.assertEqual(
                current,
                baseline,
                f"alias {name!r} must resolve to the same config as "
                f"{V5I4_ALIASES[0]!r}",
            )


class V5i4RunTagAndInitialOpponentConsistencyTests(unittest.TestCase):
    """Two logging inconsistencies the user flagged in the v5i4 launch log:

    1. The run_tag contained ``_2m_`` but the trainer reported 1,000,000
       total timesteps. v5_strict_summer / v5i1 / v5i2 / v5i3 inherited
       a misleading ``_2m_`` suffix from v4i1 without ever overriding
       ``total_timesteps`` from its 1M default. v5i4 corrects the tag.
    2. The first telemetry slice contained an OP3 entry even though the
       audit banner correctly reported the configured pool as
       ``("OP5", "OP6", "OP7")``. The cause was
       ``_resolve_initial_opponent_and_phase`` using
       ``cfg.fixed_opponent_tag`` (default ``"OP3"``) as the seeding
       opponent for the first env reset, regardless of the configured
       pool. The fix falls back to the first pool entry when the legacy
       ``fixed_opponent_tag`` is out-of-pool.
    """

    def test_run_tag_advertises_actual_total_timesteps_budget(self) -> None:
        cfg = _resolved("v5i4")
        # Default PPOConfig budget; no v5* preset overrides it.
        self.assertEqual(int(cfg.total_timesteps), 1_000_000)
        # Tag must agree with the budget so reviewers do not have to
        # cross-check the printed "Total timesteps" line against the tag.
        self.assertIn("_1m_", cfg.run_tag)
        self.assertNotIn(
            "_2m_",
            cfg.run_tag,
            "v5i4 run_tag must not carry the misleading _2m_ suffix "
            "the v4i1 / v5_strict_summer chain inherited.",
        )

    def test_initial_opponent_falls_back_to_pool_first_entry(self) -> None:
        """In pool mode, the seeding opponent must come from the pool.

        v5i4 inherits ``fixed_opponent_tag = "OP3"`` from the
        plan-faithful base, but ``opponent_pool = ("OP5", "OP6", "OP7")``.
        The first env reset must NOT use OP3, otherwise the first
        telemetry slice contradicts the audit banner.
        """
        cfg = _resolved("v5i4")
        self.assertEqual(str(cfg.fixed_opponent_tag).upper(), "OP3")
        self.assertNotIn("OP3", cfg.opponent_pool)
        _curr, _phase, tag = _resolve_initial_opponent_and_phase(cfg, max_agents=4)
        self.assertIn(
            tag,
            cfg.opponent_pool,
            "initial_opponent_tag must come from the configured pool",
        )
        self.assertEqual(tag, "OP5")

    def test_initial_opponent_respects_explicit_in_pool_fixed_tag(self) -> None:
        """A user who explicitly sets ``fixed_opponent_tag`` to an in-pool
        opponent should still win -- the fallback only fires when the
        legacy default ``OP3`` is out-of-pool."""
        cfg = _resolved("v5i4")
        cfg.fixed_opponent_tag = "OP6"
        _curr, _phase, tag = _resolve_initial_opponent_and_phase(cfg, max_agents=4)
        self.assertEqual(tag, "OP6")

    def test_initial_opponent_unchanged_for_fixed_mode_presets(self) -> None:
        """For ``FIXED_OPPONENT`` mode without ``opponent_randomize``,
        the legacy behavior must be preserved bit-for-bit."""
        cfg = PPOConfig()  # defaults: mode=FIXED_OPPONENT, fixed_opponent_tag="OP3", opponent_randomize=False
        self.assertFalse(bool(cfg.opponent_randomize))
        _curr, _phase, tag = _resolve_initial_opponent_and_phase(cfg, max_agents=4)
        self.assertEqual(tag, "OP3")


class V5i4PaperFaithfulAuditBannerTests(unittest.TestCase):
    """The launch-time audit banner emits the v5i4 invariant block when
    the run is configured per the v5i4 contract, and stays silent for
    non-v5i4 presets."""

    def _capture(self, cfg: PPOConfig) -> str:
        buf = io.StringIO()
        with redirect_stdout(buf):
            _maybe_print_paper_faithful_audit(cfg)
        return buf.getvalue()

    def test_audit_banner_fires_for_v5i4(self) -> None:
        cfg = _resolved("v5i4")
        out = self._capture(cfg)
        self.assertIn("v5i4 paper-faithful audit", out)
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
        # No warning lines should fire for a correctly-configured v5i4.
        self.assertNotIn("v5i4 audit WARNING", out)

    def test_audit_banner_silent_for_v5_strict_summer(self) -> None:
        cfg = _resolved("v5_strict_summer")
        out = self._capture(cfg)
        self.assertEqual(out, "")

    def test_audit_banner_warns_on_missing_strategy_ppo_coef(self) -> None:
        cfg = _resolved("v5i4")
        cfg.latent_strategy_ppo_coef = 0.0
        out = self._capture(cfg)
        self.assertIn("v5i4 audit WARNING", out)
        self.assertIn("latent_strategy_ppo_coef <= 0", out)

    def test_audit_banner_warns_on_dedicated_router_optimizer(self) -> None:
        cfg = _resolved("v5i4")
        cfg.latent_episode_strategy_lr = 5e-3
        out = self._capture(cfg)
        self.assertIn("v5i4 audit WARNING", out)
        self.assertIn("latent_episode_strategy_lr is set", out)

    def test_audit_banner_warns_on_film(self) -> None:
        cfg = _resolved("v5i4")
        cfg.enable_actor_z_film = True
        out = self._capture(cfg)
        self.assertIn("v5i4 audit WARNING", out)
        self.assertIn("actor-z pathway is not concat-only", out)


if __name__ == "__main__":
    unittest.main()
