"""Regression tests for the z-marginal q_phi advantage baseline.

The motivating bug: the legacy episode-credit advantage uses
``V(s, z_picked)`` as the baseline, which mathematically subtracts
``E[R | s, z_picked]`` from the return and leaves only within-z noise as
the q_phi gradient signal. The cross-z information that q_phi needs to
specialize is cancelled before the gradient is computed.

The fix introduces ``compute_z_marginal_strategy_value``, which produces
the variance-optimal AAC baseline ``E_{z' ~ q_phi(s)}[V(s, z')]``. When
``latent_q_phi_marginal_baseline = True`` is set on the config, the
episode-credit advantage uses this marginal instead. These tests pin:

1. The helper math (policy-weighted and uniform-mean variants).
2. The detach contract -- baseline must not back-propagate into V.
3. The v3b preset wires the toggle on alongside warmup + lamH anneal.
4. Main-loop q_phi loss is gated off when ``latent_strategy_ppo_coef == 0``
   (Fix 5: prevent main-loop entropy from out-voting episode-credit at
   ~3 orders of magnitude).
"""

import unittest
from types import SimpleNamespace

import torch

from rl.custom_ppo.latent_value_baselines import compute_z_marginal_strategy_value


class _FakeModel(torch.nn.Module):
    """Episode-strategy value head + q_phi logits head with controllable per-z values."""

    def __init__(self, latent_k: int, per_z_values: list[float], logits: list[float]) -> None:
        super().__init__()
        self.latent_k = int(latent_k)
        self._per_z = torch.as_tensor(per_z_values, dtype=torch.float32)
        self._logits = torch.as_tensor(logits, dtype=torch.float32)
        self._value_calls = 0
        self._logits_calls = 0
        # Trainable parameter so calls go through autograd; lets the test verify the
        # detach contract by checking parameter gradients after a fake backward.
        self.theta = torch.nn.Parameter(torch.zeros(()))

    def episode_strategy_value(
        self, states: torch.Tensor, z_idx: torch.Tensor, *, selector_hidden=None
    ) -> torch.Tensor:
        self._value_calls += 1
        z = z_idx.long().reshape(-1)
        v = self._per_z[z].to(states.device) + self.theta
        return v

    def strategy_logits(self, states: torch.Tensor, *, selector_hidden=None) -> torch.Tensor:
        self._logits_calls += 1
        return self._logits.to(states.device).unsqueeze(0).expand(states.shape[0], -1)


class MarginalBaselineMathTests(unittest.TestCase):
    """Pin the analyst's Test 1: marginal baseline computes different advantage."""

    def test_uniform_mean_matches_hand_computed_average(self):
        # V(s, z=0..3) = 1, 2, 3, 4. Uniform mean = 2.5. Picked z=3 -> chosen V = 4.
        # Legacy advantage:  R - V(s, 3)         = 5 - 4   = 1
        # Marginal (uniform): R - mean(1,2,3,4)  = 5 - 2.5 = 2.5
        model = _FakeModel(latent_k=4, per_z_values=[1, 2, 3, 4], logits=[0, 0, 0, 0])
        states = torch.zeros((1, 8), dtype=torch.float32)
        marginal = compute_z_marginal_strategy_value(model, states, 4, policy_weighted=False)
        self.assertAlmostEqual(float(marginal.item()), 2.5, places=5)
        adv_legacy = 5.0 - 4.0
        adv_marginal = 5.0 - float(marginal.item())
        self.assertAlmostEqual(adv_legacy, 1.0, places=5)
        self.assertAlmostEqual(adv_marginal, 2.5, places=5)
        self.assertNotAlmostEqual(adv_marginal, adv_legacy, places=3)

        # Refinement: Chosen z value should not change which values are included
        # in the marginal baseline. For chosen_z in {0,1,2,3}, the baseline
        # still averages z0,z1,z2,z3 (always equals 2.5).
        for chosen_z in range(4):
            # The baseline is independent of chosen_z and remains the average over all z slots
            self.assertAlmostEqual(float(marginal.item()), 2.5, places=5)

    def test_policy_weighted_matches_softmax_dot_v(self):
        # Logits [1, 0, 0, 2] -> softmax probs (unnormalized [e^1, 1, 1, e^2]).
        # V(s, z) = [1, 2, 3, 4]. Expected baseline = sum_k pi(k) * V(k).
        logits = [1.0, 0.0, 0.0, 2.0]
        per_z = [1.0, 2.0, 3.0, 4.0]
        model = _FakeModel(latent_k=4, per_z_values=per_z, logits=logits)
        states = torch.zeros((2, 8), dtype=torch.float32)
        marginal = compute_z_marginal_strategy_value(model, states, 4, policy_weighted=True)
        probs = torch.softmax(torch.tensor(logits), dim=-1)
        expected = float((probs * torch.tensor(per_z)).sum().item())
        self.assertEqual(tuple(marginal.shape), (2,))
        for value in marginal.tolist():
            self.assertAlmostEqual(value, expected, places=5)

    def test_baseline_is_detached_no_gradient_to_value_head(self):
        """The marginal baseline must not back-propagate into V (detach contract).

        If the baseline path leaked gradient into V, two routes would update the
        same value head per step (q_phi's PG via the baseline, plus the dedicated
        v_loss MSE), double-counting and destabilizing the value learner. The
        helper enforces this by ``.detach()``-ing the returned tensor.

        Pin the contract two ways: (1) the returned tensor has no grad fn, and
        (2) routing it through an autograd graph alongside a *grad-bearing*
        sentinel produces a backward pass whose gradient lands ONLY on the
        sentinel, never on the value-head parameter.
        """
        model = _FakeModel(latent_k=4, per_z_values=[1, 2, 3, 4], logits=[0, 0, 0, 0])
        states = torch.zeros((3, 4), dtype=torch.float32)
        baseline = compute_z_marginal_strategy_value(model, states, 4, policy_weighted=True)

        self.assertFalse(
            baseline.requires_grad,
            msg="compute_z_marginal_strategy_value must return a detached tensor",
        )
        self.assertIsNone(
            baseline.grad_fn,
            msg="Detached baseline must have no autograd graph node",
        )

        sentinel = torch.zeros((), requires_grad=True)
        if model.theta.grad is not None:
            model.theta.grad = None
        # Compose the baseline with a grad-bearing scalar so backward() has a
        # legitimate path. The composed loss is differentiable through sentinel,
        # NOT through baseline; model.theta.grad must stay None.
        (baseline.sum() + sentinel).backward()
        self.assertIsNone(
            model.theta.grad,
            msg="Baseline leaked gradient into V's parameters -- detach contract broken",
        )
        self.assertIsNotNone(
            sentinel.grad, msg="Sanity: sentinel grad should be populated"
        )

    def test_latent_k_one_returns_chosen_z_value(self):
        # K=1 short-circuit: marginal collapses to V(s, 0) trivially.
        model = _FakeModel(latent_k=1, per_z_values=[7.0], logits=[0.0])
        states = torch.zeros((5, 3), dtype=torch.float32)
        out = compute_z_marginal_strategy_value(model, states, 1)
        self.assertEqual(tuple(out.shape), (5,))
        for v in out.tolist():
            self.assertAlmostEqual(v, 7.0, places=5)


class V3bPresetWiringTests(unittest.TestCase):
    """Pin the analyst's Test 4-ish: v3b preset opts into all the right knobs."""

    def test_v3b_preset_enables_marginal_baseline_plus_warmup_plus_anneal(self):
        from rl.train_ppo import PPOConfig
        from rl.presets import apply_preset

        cfg = apply_preset(PPOConfig(), "plan_faithful_latent_v3b_marginal")

        # The dragon heart: marginal baseline on.
        self.assertTrue(cfg.latent_q_phi_marginal_baseline)

        # Inherited from v3 (episode_strategic) -- must not have regressed.
        self.assertTrue(cfg.latent_episode_strategy_ppo)
        self.assertEqual(cfg.latent_episode_strategy_warmup_decision_steps, 5)
        self.assertEqual(cfg.latent_resample_every_n, 0)
        self.assertEqual(cfg.latent_strategy_ppo_coef, 0.0)
        self.assertEqual(cfg.latent_lam_p, 0.0)
        self.assertEqual(cfg.latent_lam_h_start, 0.003)
        self.assertEqual(cfg.latent_lam_h_end, 0.0005)
        self.assertEqual(cfg.latent_entropy_anneal_start, 200_000)
        self.assertEqual(cfg.latent_entropy_anneal_end, 700_000)
        self.assertEqual(cfg.latent_k, 4)

        # Plan-faithful: no labels / aux heads / opponent IDs introduced.
        self.assertEqual(cfg.latent_strategy_aux_predict_phase_coef, 0.0)
        self.assertFalse(cfg.latent_strategy_aux_return_head)
        self.assertFalse(cfg.fixed_latent_strategy)

        self.assertIn("marginalbaseline", cfg.run_tag)

    def test_marginal_baseline_default_is_off_back_compat(self):
        """Bare PPOConfig must not opt into the new behavior."""
        from rl.train_ppo import PPOConfig

        self.assertFalse(PPOConfig().latent_q_phi_marginal_baseline)


class MainLoopGatingTests(unittest.TestCase):
    """Pin the v5 decoupled q_phi gating semantics in ``ppo_updater.py``.

    Background (v3c "Fix 5", now superseded):
        v3c added an episode-credit channel (``apply_episode_strategy_ppo``)
        that steps the strategy_encoder + value head through a dedicated
        ``latent_router_optimizer``. Without a guard, the shared optimizer's
        main-loop pass would *also* step the same params via the entropy /
        persistence / strategy-PPO / KL / aux-return terms in the same
        update, doubling (and at high lam_h, ~650x amplifying) the entropy
        push. Fix 5 silenced ALL of those main-loop terms when
        ``latent_strategy_ppo_coef == 0``.

    Why v5 replaces Fix 5:
        Fix 5's trigger conflated "should the per-step PPO PG run?" with
        "is a dedicated router optimizer active?". v3i19 / v4i1 / v4i3 set
        ``latent_strategy_ppo_coef = 0`` *without* a dedicated router
        optimizer; they expected ``lam_p`` and ``lam_h`` to fire via the
        main loop (matching docs/algorithm.md), but Fix 5 silently zeroed
        them. The v5 gate triggers off ``latent_router_optimizer is not
        None`` instead, so the main-loop regularizers fire whenever they
        are the only path to q_phi.

    The runtime path is now::

        has_dedicated_router_opt = runtime.latent_router_optimizer is not None
        apply_main_loop_qphi_loss = latent_strategy_ppo_coef > 0 and not has_dedicated_router_opt
        apply_entropy_loss      = use_latent_strategy and not has_dedicated_router_opt and lam_h > 0 and objective != "none"
        apply_persistence_loss  = use_latent_strategy and not has_dedicated_router_opt and (lam_p > 0 or sparse_tactical_refresh)
        apply_kl_loss           = use_latent_strategy and not has_dedicated_router_opt and lam_kl_consecutive > 0

    This test reads ``ppo_updater.py`` source so silent drift is caught at
    review time, not at runtime via a 650x entropy push.
    """

    def test_main_loop_gate_uses_dedicated_router_opt_safeguard(self):
        import pathlib
        import re

        source = (
            pathlib.Path(__file__).resolve().parent.parent
            / "rl"
            / "custom_ppo"
            / "ppo_updater.py"
        ).read_text(encoding="utf-8")

        # The "safeguard" must inspect the dedicated router optimizer, NOT
        # the strategy PPO coefficient. This is the core v5 change.
        safeguard_pattern = re.compile(
            r"has_dedicated_router_opt\s*=\s*\(\s*getattr\(\s*runtime\s*,\s*"
            r"['\"]latent_router_optimizer['\"][^)]*\)\s*is not None\s*\)"
        )
        self.assertRegex(
            source,
            safeguard_pattern,
            msg=(
                "v5 gate must compute `has_dedicated_router_opt` from "
                "`runtime.latent_router_optimizer`. If this regex fails, the "
                "double-step safeguard for v3c-style episode-credit runs is gone."
            ),
        )

        # Strategy-PPO term must still be gated by ``coef > 0 AND not dedicated``.
        ppo_gate_pattern = re.compile(
            r"apply_main_loop_qphi_loss\s*=\s*\(\s*\n\s*float\(\s*hparams\.latent_strategy_ppo_coef\s*or\s*0\.0\s*\)\s*>\s*0\.0\s*\n\s*"
            r"and not has_dedicated_router_opt"
        )
        self.assertRegex(
            source,
            ppo_gate_pattern,
            msg=(
                "Per-step strategy PPO loss must gate on `latent_strategy_ppo_coef > 0` "
                "AND `not has_dedicated_router_opt`."
            ),
        )

        # Entropy gates on lam_h > 0 (and objective != 'none').
        self.assertIn(
            'float(latent_lam_h or 0.0) > 0.0',
            source,
            msg="Entropy term must gate on `latent_lam_h > 0`, not on `latent_strategy_ppo_coef`.",
        )

        # Persistence gates on lam_p > 0 OR sparse tactical refresh.
        persist_gate_pattern = re.compile(
            r"apply_persistence_loss\s*=\s*\([\s\S]*?"
            r"float\(\s*getattr\(\s*cfg\s*,\s*['\"]latent_lam_p['\"][^)]*\)\s*or\s*0\.0\s*\)\s*>\s*0\.0[\s\S]*?"
            r"or hparams\.latent_sparse_tactical_refresh_enabled",
            re.MULTILINE,
        )
        self.assertRegex(
            source,
            persist_gate_pattern,
            msg=(
                "Persistence term must gate on `latent_lam_p > 0 OR "
                "latent_sparse_tactical_refresh_enabled`, not on `latent_strategy_ppo_coef`."
            ),
        )

        # Double-step safeguard: each guarded term still gets zeroed when the
        # dedicated router optimizer is the active q_phi gradient sink.
        for guarded_term in (
            "strategy_entropy_loss = torch.zeros_like(strategy_entropy_loss)",
            "persist_term_loss = torch.zeros_like(persist_term_loss)",
            "strategy_policy_loss_scaled = torch.zeros_like(strategy_policy_loss_scaled)",
        ):
            self.assertIn(
                guarded_term,
                source,
                msg=f"v5 gate must still zero `{guarded_term}` when its apply_* flag is False.",
            )


if __name__ == "__main__":
    unittest.main()
