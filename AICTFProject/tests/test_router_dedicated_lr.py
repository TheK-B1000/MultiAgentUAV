"""Regression tests for v3c: q_phi inner-epoch loop + dedicated router optimizer.

The motivating diagnosis (post-v3b): the marginal baseline gave q_phi a real
non-zero gradient signal (``episode_credit_grad_norm`` ~0.005-0.027 from
update 1), but cumulative logit change over a 1M-step run was only ~10^-5 --
five orders of magnitude short of the ~ln(2)=0.7 needed to break q_phi off
uniform for K=4 strategies. Two compounding constraints on the effective
router step:

  (1) ``apply_episode_strategy_ppo`` ran ONE backward step per rollout (vs
      the actor's 6-8 PPO inner epochs).
  (2) The shared optimizer's LR (1.35e-4 for 4v4) was actor-tuned and ~37x
      too small for q_phi's clean-but-small gradient.

v3c lifts both with config-only knobs:

  - ``latent_episode_strategy_n_epochs`` (default 1, set to 6 in v3c preset)
  - ``latent_episode_strategy_lr``       (default None, set to 5e-3 in v3c
                                          preset, builds a dedicated AdamW
                                          for strategy_encoder +
                                          episode_strategy_value_head)

These tests pin:

1. Bare-config defaults preserve legacy behavior (n_epochs=1, lr=None).
2. v3c preset wires both knobs + raises the entropy floor to 0.001.
3. TrainerHyperparams plumbs both knobs through ``cfg -> hparams`` correctly.
4. The runtime path in ``apply_episode_strategy_ppo`` actually loops on
   ``latent_episode_strategy_n_epochs`` and routes through ``router_optimizer``
   (the dedicated one when present, else the shared one). Pinned via
   source-text inspection -- the only safe way to assert this without
   instantiating a full trainer.
5. The dedicated-optimizer build in ``trainer_optimizers.py`` is gated on
   ``latent_episode_strategy_lr is not None`` AND on the latent strategy being
   enabled (no surprise optimizer when latent is off or fixed_z is on).
"""

from __future__ import annotations

import pathlib
import re
import unittest


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_EPISODE_CREDIT_SRC = (
    _REPO_ROOT / "rl" / "custom_ppo" / "latent" / "credit" / "episode" / "manager.py"
).read_text(encoding="utf-8")
_ROUTER_PPO_SRC = (
    _REPO_ROOT / "rl" / "custom_ppo" / "latent" / "optimization" / "router_ppo.py"
).read_text(encoding="utf-8")
_ROUTER_REGISTRY_SRC = (
    _REPO_ROOT / "rl" / "custom_ppo" / "latent" / "optimization" / "router_registry.py"
).read_text(encoding="utf-8")
_ROUTER_STEPPER_SRC = (
    _REPO_ROOT / "rl" / "custom_ppo" / "latent" / "optimization" / "router_stepper.py"
).read_text(encoding="utf-8")
_TRAINER_SRC = (_REPO_ROOT / "rl" / "custom_ppo" / "trainer.py").read_text(encoding="utf-8")
_OPTIMIZER_BUNDLE_SRC = (
    _REPO_ROOT / "rl" / "custom_ppo" / "trainer_optimizers.py"
).read_text(encoding="utf-8")


class V3cConfigDefaultsTests(unittest.TestCase):
    """Back-compat: a bare PPOConfig must NOT change router-update behavior."""

    def test_n_epochs_default_is_one(self):
        from rl.train_ppo import PPOConfig

        cfg = PPOConfig()
        self.assertEqual(
            int(cfg.latent_episode_strategy_n_epochs),
            1,
            msg=(
                "Default n_epochs must be 1 -- raising it silently would 6x "
                "the q_phi update cost for every existing preset."
            ),
        )

    def test_lr_default_is_none(self):
        from rl.train_ppo import PPOConfig

        cfg = PPOConfig()
        self.assertIsNone(
            cfg.latent_episode_strategy_lr,
            msg=(
                "Default LR must be None -- existing presets must keep using "
                "the shared optimizer with no dedicated router LR."
            ),
        )


class V3cPresetWiringTests(unittest.TestCase):
    """Pin the v3c preset opts into all three router-strength knobs."""

    def _v3c_cfg(self):
        from rl.train_ppo import PPOConfig
        from rl.presets import apply_preset

        return apply_preset(PPOConfig(), "plan_faithful_latent_v3c_router_lr")

    def test_v3c_sets_n_epochs_six(self):
        cfg = self._v3c_cfg()
        self.assertEqual(cfg.latent_episode_strategy_n_epochs, 6)

    def test_v3c_sets_router_lr_5e3(self):
        cfg = self._v3c_cfg()
        self.assertIsNotNone(cfg.latent_episode_strategy_lr)
        self.assertAlmostEqual(float(cfg.latent_episode_strategy_lr), 5e-3, places=8)

    def test_v3c_raises_entropy_floor_to_1e3(self):
        # v3b annealed to 0.0005; v3c lifts the floor as collapse insurance for
        # the much larger effective router updates.
        cfg = self._v3c_cfg()
        self.assertAlmostEqual(float(cfg.latent_lam_h_end), 0.001, places=8)

    def test_v3c_inherits_v3b_marginal_baseline(self):
        # v3c is built on top of v3b; the marginal baseline must still be on.
        cfg = self._v3c_cfg()
        self.assertTrue(cfg.latent_q_phi_marginal_baseline)
        self.assertTrue(cfg.latent_episode_strategy_ppo)
        self.assertEqual(cfg.latent_episode_strategy_warmup_decision_steps, 5)
        self.assertEqual(cfg.latent_strategy_ppo_coef, 0.0)
        self.assertEqual(cfg.latent_lam_p, 0.0)
        self.assertEqual(cfg.latent_k, 4)

    def test_v3c_plan_faithful_no_labels_no_aux(self):
        cfg = self._v3c_cfg()
        self.assertEqual(cfg.latent_strategy_aux_predict_phase_coef, 0.0)
        self.assertFalse(cfg.latent_strategy_aux_return_head)
        self.assertFalse(cfg.fixed_latent_strategy)

    def test_v3c_alias_resolves(self):
        from rl.train_ppo import PPOConfig
        from rl.presets import apply_preset

        # Short alias must point to the same function.
        cfg_short = apply_preset(PPOConfig(), "latent_v3c")
        cfg_long = apply_preset(PPOConfig(), "plan_faithful_latent_v3c_router_lr")
        # Compare a representative knob from each subsystem we touched.
        self.assertEqual(
            cfg_short.latent_episode_strategy_n_epochs,
            cfg_long.latent_episode_strategy_n_epochs,
        )
        self.assertEqual(
            cfg_short.latent_episode_strategy_lr,
            cfg_long.latent_episode_strategy_lr,
        )
        self.assertEqual(cfg_short.latent_lam_h_end, cfg_long.latent_lam_h_end)


_TRAINER_CONFIG_SRC = (
    _REPO_ROOT / "rl" / "custom_ppo" / "trainer_config.py"
).read_text(encoding="utf-8")


class V3cHyperparamsPlumbingTests(unittest.TestCase):
    """``cfg.latent_episode_strategy_*`` must land on ``hparams`` unchanged.

    Constructing a real TrainerHyperparams requires an env + curriculum + many
    derived knobs that aren't relevant to this regression. Source inspection
    is the cheap, equivalent contract: pin that both fields exist on the
    dataclass and that ``from_ppo_config`` reads them from cfg with the right
    normalization.
    """

    def test_dataclass_declares_n_epochs_and_lr_fields(self):
        # Dataclass field declarations -- ensures the trainer can ever see these.
        self.assertRegex(
            _TRAINER_CONFIG_SRC,
            r"latent_episode_strategy_n_epochs\s*:\s*int\b",
            msg="TrainerHyperparams must declare latent_episode_strategy_n_epochs",
        )
        self.assertRegex(
            _TRAINER_CONFIG_SRC,
            r"latent_episode_strategy_lr\s*:\s*Optional\[\s*float\s*\]",
            msg="TrainerHyperparams must declare latent_episode_strategy_lr: Optional[float]",
        )

    def test_from_ppo_config_plumbs_n_epochs_with_floor_of_one(self):
        # n_epochs must be read from cfg with int() coercion and floor of 1 --
        # otherwise n_epochs=0 silently skips the entire q_phi update.
        self.assertRegex(
            _TRAINER_CONFIG_SRC,
            r"latent_episode_strategy_n_epochs\s*=\s*max\(\s*\n?\s*1\s*,\s*"
            r"int\(\s*getattr\(\s*cfg\s*,\s*['\"]latent_episode_strategy_n_epochs['\"]\s*,\s*1\s*\)"
            r"\s*or\s*1\s*\)\s*\)",
            msg=(
                "from_ppo_config must read latent_episode_strategy_n_epochs from cfg "
                "with `max(1, int(... or 1))` to coerce 0/None/negative to 1."
            ),
        )

    def test_from_ppo_config_plumbs_lr_with_none_passthrough(self):
        # LR must convert to float when present and stay None when absent --
        # the dedicated optimizer init in trainer.py specifically tests `is not None`.
        self.assertRegex(
            _TRAINER_CONFIG_SRC,
            r"latent_episode_strategy_lr\s*=\s*\(\s*\n?\s*"
            r"float\(\s*getattr\(\s*cfg\s*,\s*['\"]latent_episode_strategy_lr['\"]\s*,\s*None\s*\)\s*\)"
            r"\s*\n?\s*if\s+getattr\(\s*cfg\s*,\s*['\"]latent_episode_strategy_lr['\"]\s*,\s*None\s*\)\s+is\s+not\s+None"
            r"\s*\n?\s*else\s+None",
            msg=(
                "from_ppo_config must coerce cfg.latent_episode_strategy_lr to float "
                "when set and pass through None otherwise."
            ),
        )


class V3cRuntimePathTests(unittest.TestCase):
    """Pin the runtime wiring without booting the trainer.

    The full ``apply_episode_strategy_ppo`` requires a real model, optimizer,
    rollout buffer, and CUDA-or-CPU device. Instead of stubbing all of that,
    we pin the invariants that matter via source-text inspection -- any
    regression that breaks these would silently revert v3c to v3b behavior.
    """

    def test_apply_episode_strategy_ppo_has_inner_epoch_loop(self):
        self.assertRegex(
            _EPISODE_CREDIT_SRC,
            r"n_epochs\s*=\s*max\(\s*1\s*,\s*int\(\s*getattr\(\s*trainer\s*,\s*"
            r"['\"]latent_episode_strategy_n_epochs['\"]\s*,\s*1\s*\)\s*or\s*1\s*\)\s*\)",
            msg=(
                "apply_episode_strategy_ppo must derive n_epochs from "
                "trainer.latent_episode_strategy_n_epochs (with floor of 1). "
                "If this regex fails the inner loop was removed or rewired."
            ),
        )
        self.assertRegex(
            _ROUTER_PPO_SRC,
            r"for\s+epoch\s+in\s+range\(\s*max\(\s*1\s*,\s*int\(\s*config\.epochs\s*\)\s*\)\s*\)\s*:",
            msg=(
                "Inner-epoch loop missing in RouterPPOEngine.run. "
                "Without it, latent_episode_strategy_n_epochs > 1 has no effect."
            ),
        )

    def test_apply_episode_strategy_ppo_routes_through_router_optimizer(self):
        self.assertRegex(
            _ROUTER_REGISTRY_SRC,
            r"getattr\(\s*trainer\s*,\s*['\"]router_optimizer['\"]\s*,\s*None\s*\)\s*or\s*getattr\(\s*"
            r"trainer\s*,\s*['\"]latent_router_optimizer['\"]\s*,\s*None\s*\)",
            msg=(
                "LatentOptimizerRegistry must prefer trainer.router_optimizer, "
                "then trainer.latent_router_optimizer. "
                "Without this, the v3c dedicated LR has no effect."
            ),
        )
        self.assertIn("self.registry.zero_grad(set_to_none=True)", _ROUTER_STEPPER_SRC)
        self.assertIn("self.registry.step()", _ROUTER_STEPPER_SRC)

    def test_clip_grad_norm_scope_is_router_params_when_dedicated(self):
        """When using the dedicated optimizer, clip only its params -- not the whole model."""
        self.assertRegex(
            _ROUTER_STEPPER_SRC,
            r"clip_grad_norm_\(\s*self\.registry\.router_parameters",
            msg=(
                "RouterOptimizerStepper must clip only router optimizer params, "
                "not trainer.model.parameters()."
            ),
        )


class V3cTrainerInitTests(unittest.TestCase):
    """Pin that the dedicated optimizer is built iff LR is set AND latent is on."""

    def test_dedicated_optimizer_gated_on_lr_not_none(self):
        self.assertRegex(
            _OPTIMIZER_BUNDLE_SRC,
            r"if\s*\(\s*\n?\s*hparams\.latent_episode_strategy_lr\s+is\s+None",
            msg=(
                "TrainerOptimizerBundle must gate the dedicated router optimizer on "
                "hparams.latent_episode_strategy_lr is not None."
            ),
        )

    def test_dedicated_optimizer_gated_on_latent_enabled(self):
        self.assertIn(
            "hparams.use_latent_strategy", _OPTIMIZER_BUNDLE_SRC,
            msg="Dedicated optimizer init must guard on use_latent_strategy",
        )
        self.assertIn(
            "hparams.fixed_latent_strategy", _OPTIMIZER_BUNDLE_SRC,
            msg="Dedicated optimizer init must guard on not fixed_latent_strategy",
        )

    def test_dedicated_optimizer_targets_strategy_encoder_and_value_head(self):
        self.assertIn("strategy_encoder", _OPTIMIZER_BUNDLE_SRC)
        self.assertIn("episode_strategy_value_head", _OPTIMIZER_BUNDLE_SRC)
        self.assertRegex(
            _OPTIMIZER_BUNDLE_SRC,
            r"torch\.optim\.AdamW\(\s*\n?\s*router_params",
            msg="Dedicated optimizer must be AdamW over the router params list",
        )

    def test_dedicated_optimizer_attribute_initialized_to_none(self):
        self.assertIn('"latent_router_optimizer"', _TRAINER_SRC)
        self.assertIn("optimizers.latent_router_optimizer", _TRAINER_SRC)
        self.assertRegex(
            _OPTIMIZER_BUNDLE_SRC,
            r"def\s+latent_router_optimizer\s*\(\s*self\s*\)\s*->",
            msg=(
                "TrainerOptimizerBundle.latent_router_optimizer must exist so "
                "trainer.__getattr__ can forward legacy lookups safely."
            ),
        )


if __name__ == "__main__":
    unittest.main()
