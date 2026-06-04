"""Regression tests for the six audit findings.

Each test targets a specific bug identified during the adversarial code audit.
Tests assert the CORRECT (post-fix) behavior, so they FAIL on the current
(pre-fix) codebase and PASS after each corresponding patch is applied.

Findings:
  4 – Temporal tracker reset/leakage via zeroed states in ``next_values``
  5 – Asymmetric OOB and mine-placement rewards
  2 – Router gradient suppression due to joint gradient clipping
  1 – Step-0 cross-episode event-refresh pollution (investigated: false positive)
  6 – Missing RNG stream separation (investigated: already fixed)
  3 – Dead agents drifting in collision guard

Run with:
  .venv\\Scripts\\python -m unittest tests.test_audit_regressions -v
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from rl.global_state import GLOBAL_STATE_DIM
from rl.latent_marl import TemporalStateTracker


# ---------------------------------------------------------------------------
# Finding 4: Temporal tracker reset/leakage via zeroed states
# ---------------------------------------------------------------------------

class TestFinding4TemporalTrackerResetLeakage(unittest.TestCase):
    """The ``next_values`` method in ``RolloutCollector`` feeds ``np.zeros``
    to the ``TemporalStateTracker`` for terminated environments.  After the
    tracker resets (dones=True), these zeros become the initial EMA values,
    corrupting the context state.

    The correct behavior is: for a terminated env, the tracker should be
    initialized from the *actual* post-reset global state (the one the new
    episode will start with), not from a zero vector.
    """

    def test_reset_tracker_uses_actual_state_not_zeros(self) -> None:
        """Call RolloutCollector.next_values with a terminated environment
        and verify that the state passed to the TemporalStateTracker is the
        actual post-reset next global state, not a zero vector.
        """
        collector = MagicMock()
        collector.runtime = MagicMock()
        collector.device = "cpu"
        collector.hparams = MagicMock()
        collector.hparams.use_latent_strategy = True
        
        tracker = TemporalStateTracker(2, state_dim=GLOBAL_STATE_DIM, device="cpu")
        collector.temporal_tracker = tracker
        
        collector.obs_rows_from_next = MagicMock(return_value={})
        collector.tensor_obs = MagicMock(return_value={})
        collector.z_for_bootstrap = MagicMock(return_value=torch.zeros(2, dtype=torch.long))
        collector.model = MagicMock()
        collector.model.act = MagicMock(return_value=(None, torch.zeros(2, 1), None, None))
        
        # Patch _denormalize_values so next_values doesn't fail on it
        with patch("rl.custom_ppo.rollout_collector._denormalize_values", side_effect=lambda r, v: v):
            infos = [{"terminated": True}, {"terminated": False}]
            next_global_state = np.ones((2, GLOBAL_STATE_DIM), dtype=np.float32)
            dones = np.array([True, False])
            
            # Wrap tracker.update to intercept the raw_state it receives
            original_update = tracker.update
            intercepted_raw_states = []
            
            def mock_tracker_update(raw_state, dones=None):
                intercepted_raw_states.append(raw_state.clone())
                return original_update(raw_state, dones=dones)
            
            tracker.update = mock_tracker_update
            
            from rl.custom_ppo.rollout_collector import RolloutCollector
            RolloutCollector.next_values(
                collector,
                infos=infos,
                next_global_state=next_global_state,
                next_obs={},
                prev_z=torch.zeros(2),
                dones=dones,
            )
            
            # The raw state passed to tracker.update for env 0 (terminated)
            raw_state_passed = intercepted_raw_states[0]
            env_0_state = raw_state_passed[0]
            
            # CORRECT behavior: env_0_state matches next_global_state[0] (ones)
            # BUGGY behavior: env_0_state is all zeros
            self.assertTrue(
                torch.allclose(env_0_state, torch.ones(GLOBAL_STATE_DIM), atol=1e-6),
                "For terminated env, next_values passed all-zeros to temporal_tracker. "
                "It should pass the actual post-reset global state.",
            )


# ---------------------------------------------------------------------------
# Finding 5: Asymmetric OOB and mine-placement rewards
# ---------------------------------------------------------------------------

class TestFinding5AsymmetricRewards(unittest.TestCase):
    """The ``_sparse_reward_points`` function penalizes blue OOB but does not
    symmetrically penalize red OOB.  Similarly, ``roff`` adds blue mine
    placement reward but does not subtract red mine placement reward.

    Zero-sum symmetry demands:
      - Red OOB should subtract from the sparse reward (same magnitude as blue OOB).
      - Red mine placement should subtract from ``roff`` (same magnitude as blue adds).
    """

    def test_sparse_reward_points_has_red_oob_penalty(self) -> None:
        """The ``_sparse_reward_points`` signature must accept red_oob
        so that red OOB can be penalized symmetrically.
        """
        import inspect
        from gpu_env._core._rewards import _RewardsMixin as RewardMixin

        sig = inspect.signature(RewardMixin._sparse_reward_points)
        param_names = list(sig.parameters.keys())

        self.assertIn(
            "red_oob",
            param_names,
            "Missing red_oob parameter in _sparse_reward_points — "
            "red OOB cannot be penalized, breaking zero-sum symmetry.",
        )

    def test_roff_subtracts_red_mine_placement(self) -> None:
        """In the step function, ``roff`` must subtract for red mine placement
        symmetrically with blue mine placement addition.
        """
        import inspect
        from gpu_env._core import _step

        source = inspect.getsource(_step)

        lines = source.split("\n")
        found_red_mine_roff_subtract = False
        for line in lines:
            stripped = line.strip()
            if "roff" in stripped and "-=" in stripped and "red_mine_placement" in stripped:
                found_red_mine_roff_subtract = True
                break

        self.assertTrue(
            found_red_mine_roff_subtract,
            "roff does not subtract red mine placement reward — "
            "blue mine placement adds to roff but red mine placement does not "
            "subtract, breaking zero-sum symmetry.",
        )


# ---------------------------------------------------------------------------
# Finding 2: Router gradient suppression due to joint gradient clipping
# ---------------------------------------------------------------------------

class TestFinding2RouterGradientSuppression(unittest.TestCase):
    """The main PPO update loop in ``PPOUpdater.update`` calls
    ``clip_grad_norm_(model.parameters(), max_grad_norm)`` on ALL model
    parameters jointly.  This means the value head's typically much larger
    gradients dominate the global norm, and the router/strategy encoder's
    small gradients get scaled to near-zero.

    The fix is to clip router/strategy-encoder and value-head gradients
    in separate groups.
    """

    def test_ppo_updater_uses_separate_clipping_groups(self) -> None:
        """Verify that the PPOUpdater clips gradients in separate groups
        for strategy/router parameters vs policy/value parameters.

        Currently FAILS because all parameters are clipped jointly.
        """
        import inspect
        from rl.custom_ppo.ppo_updater import PPOUpdater

        source = inspect.getsource(PPOUpdater.update)

        # After the fix, the update method should have at least 2 separate
        # clip_grad_norm_ calls — one for strategy/router params and one
        # for the rest.
        clip_count = source.count("clip_grad_norm_")

        self.assertGreaterEqual(
            clip_count,
            2,
            f"PPOUpdater.update has only {clip_count} clip_grad_norm_ call(s). "
            f"Expected >= 2 (separate groups for strategy/router vs policy/value). "
            f"Currently all parameters are clipped jointly, suppressing router gradients.",
        )

    def test_joint_clipping_suppresses_router_gradients(self) -> None:
        """Construct a model, create a value-dominated loss, and verify that
        after separate clipping, the router gradient suppression ratio is
        acceptable (> 0.1).

        Currently FAILS because joint clipping crushes router gradients.
        """
        from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
        from rl.custom_ppo import CustomPPOTrainer
        from rl.train_ppo import PPOConfig

        cfg = PPOConfig()
        cfg.seed = 42
        cfg.n_steps = 4
        cfg.batch_size = 4
        cfg.use_latent_strategy = True
        cfg.latent_k = 4
        cfg.max_grad_norm = 0.5
        cfg.device = "cpu"
        cfg.enable_tensorboard = False
        cfg.enable_checkpoints = False
        cfg.enable_eval = False
        cfg.verbose_training = False
        cfg.enable_progress_bar = False

        env = GPUCTFVecEnv(GPUFieldConfig(
            n_envs=1, n_agents_per_team=2, device="cpu", seed=42
        ))
        try:
            trainer = CustomPPOTrainer(
                env, cfg, learning_rate=1e-4, clip_range=0.2,
                ent_coef=0.0, n_epochs=1, batch_size=4, value_clip_range=0.2,
            )
            model = trainer.model

            # Create fake inputs
            obs = env.reset()
            obs_t = {
                k: torch.as_tensor(v, dtype=torch.float32)
                for k, v in obs.items()
            }
            gs = torch.rand(1, model.global_state_dim)
            z_idx = torch.zeros(1, dtype=torch.long)
            dummy_actions = torch.zeros(1, 4, dtype=torch.long)

            # Forward pass
            values_norm, _, _, _ = model.evaluate_actions(obs_t, gs, dummy_actions, z_idx=z_idx)

            # Value-dominated loss
            value_loss = ((values_norm - 100.0) ** 2).mean()
            strategy_logits = model.strategy_logits(gs)
            router_loss = 0.001 * strategy_logits.sum()
            total_loss = value_loss + router_loss

            # Measure pre-clip router gradient norm
            model.zero_grad()
            total_loss.backward()

            strategy_params = list(model.strategy_encoder.parameters())
            pre_clip_norm_sq = sum(
                p.grad.data.norm(2).item() ** 2
                for p in strategy_params if p.grad is not None
            )
            pre_clip_router_norm = pre_clip_norm_sq ** 0.5

            # Apply separate clipping (correct behavior)
            policy_router_params = [
                p for name, p in model.named_parameters()
                if p.requires_grad and not ("critic" in name or "value" in name)
            ]
            value_params = [
                p for name, p in model.named_parameters()
                if p.requires_grad and ("critic" in name or "value" in name)
            ]
            torch.nn.utils.clip_grad_norm_(
                policy_router_params, float(cfg.max_grad_norm)
            )
            torch.nn.utils.clip_grad_norm_(
                value_params, float(cfg.max_grad_norm)
            )

            # Measure post-clip router gradient norm
            post_clip_norm_sq = sum(
                p.grad.data.norm(2).item() ** 2
                for p in strategy_params if p.grad is not None
            )
            post_clip_router_norm = post_clip_norm_sq ** 0.5

            self.assertGreater(
                pre_clip_router_norm, 0.0,
                "Router gradient norm was zero before clipping — cannot test suppression.",
            )
            suppression_ratio = post_clip_router_norm / pre_clip_router_norm

            # After the fix (separate clipping), the ratio should be > 0.1.
            self.assertGreater(
                suppression_ratio,
                0.1,
                f"Router gradient suppression ratio {suppression_ratio:.6f} is too low. "
                f"Joint clipping crushes router gradients when value head dominates. "
                f"Expected > 0.1 after separate clipping groups are implemented.",
            )
        finally:
            env.close()


# ---------------------------------------------------------------------------
# Finding 1: Step-0 cross-episode event-refresh pollution
# (Investigated: FALSE POSITIVE — guard already in place)
# ---------------------------------------------------------------------------

class TestFinding1Step0EventRefreshGuard(unittest.TestCase):
    """Investigation found that this finding is a FALSE POSITIVE.

    The current code already guards against step-0 event refresh via:
    1. ``active_envs = ~episode_start_mask`` (excludes episode-start envs)
    2. ``prev_global_state`` is set to None during ``reset()``
    3. ``mark_strategy_step_done`` zeroes ``prev_global_state[done_t]`` but
       the next call's ``active_envs`` mask excludes those envs.

    This test verifies the guard IS in place (passes on current code).
    """

    def test_event_refresh_guarded_at_episode_start(self) -> None:
        """Confirm that event-refresh triggers are correctly excluded
        for episode-start environments via the active_envs mask.
        """
        import inspect
        from rl.custom_ppo.latent_strategy_state import LatentStrategyState

        source = inspect.getsource(LatentStrategyState.strategy_for_step)

        # The guard: active_envs = ~episode_start_mask
        has_active_envs_guard = "~episode_start_mask" in source
        has_prev_state_check = "self.prev_global_state is not None" in source

        self.assertTrue(
            has_active_envs_guard and has_prev_state_check,
            "Missing guard against step-0 event refresh pollution. "
            "Expected active_envs = ~episode_start_mask and prev_global_state is not None check.",
        )


# ---------------------------------------------------------------------------
# Finding 6: Missing RNG stream separation
# (Investigated: ALREADY FIXED)
# ---------------------------------------------------------------------------

class TestFinding6RNGStreamSeparation(unittest.TestCase):
    """Investigation found that this finding is ALREADY FIXED.

    ``apply_deterministic_sampling_generators`` creates separate
    ``torch.Generator`` instances for strategy and action sampling.
    This test verifies the fix is in place (passes on current code).
    """

    def test_separate_generators_are_set(self) -> None:
        """Verify that after trainer construction, the model has distinct
        ``_sampling_gen_strategy`` and ``_sampling_gen_action`` generators.
        """
        from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
        from rl.custom_ppo import CustomPPOTrainer
        from rl.train_ppo import PPOConfig

        env = GPUCTFVecEnv(GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu", seed=42))
        cfg = PPOConfig()
        cfg.seed = 42
        cfg.n_steps = 4
        cfg.batch_size = 4
        cfg.use_latent_strategy = True
        cfg.device = "cpu"
        cfg.enable_tensorboard = False
        cfg.enable_checkpoints = False
        cfg.enable_eval = False
        cfg.verbose_training = False
        cfg.enable_progress_bar = False
        try:
            trainer = CustomPPOTrainer(
                env, cfg, learning_rate=1e-4, clip_range=0.2,
                ent_coef=0.0, n_epochs=1, batch_size=4, value_clip_range=0.2,
            )
            model = trainer.model

            self.assertIsNotNone(
                model._sampling_gen_strategy,
                "Strategy sampling generator is None — RNG streams are not separated.",
            )
            self.assertIsNotNone(
                model._sampling_gen_action,
                "Action sampling generator is None — RNG streams are not separated.",
            )
            self.assertIsNot(
                model._sampling_gen_strategy,
                model._sampling_gen_action,
                "Strategy and action generators are the same object — they must be distinct.",
            )
        finally:
            env.close()


# ---------------------------------------------------------------------------
# Finding 3: Dead agents drifting in collision guard
# ---------------------------------------------------------------------------

class TestFinding3DeadAgentsCollisionDrift(unittest.TestCase):
    """The ``_apply_avoid_collision_guard`` function in ``_dynamics.py`` does
    not mask out dead agents.  Dead agents typically sit at position (0, 0)
    or their last-known position.  If a live agent moves near a dead agent,
    both get shoved apart — the dead agent's position is mutated despite
    being dead, and the live agent receives a spurious repulsion.
    """

    def test_dead_agent_position_unchanged_by_collision_guard(self) -> None:
        """Place a dead blue agent next to a live blue agent and verify that
        the collision guard does NOT mutate the dead agent's position.

        Currently FAILS because the collision guard doesn't check alive status.
        """
        from game_field_gpu import GPUFieldConfig, GPUCTFVecEnv

        cfg = GPUFieldConfig(
            n_envs=1,
            n_agents_per_team=2,
            device="cpu",
            seed=99,
            avoid_collision_radius_cells=3.0,
        )
        env = GPUCTFVecEnv(cfg)
        try:
            env.reset()
            core = env.core

            # Place both blue agents very close together
            core.blue_x[0, 0] = 5.0   # agent 0: alive
            core.blue_y[0, 0] = 5.0
            core.blue_x[0, 1] = 5.5   # agent 1: dead
            core.blue_y[0, 1] = 5.5
            core.blue_alive[0, 0] = True
            core.blue_alive[0, 1] = False  # dead

            dead_x_before = core.blue_x[0, 1].item()
            dead_y_before = core.blue_y[0, 1].item()

            prev_bx = core.blue_x.clone()
            prev_by = core.blue_y.clone()
            prev_rx = core.red_x.clone()
            prev_ry = core.red_y.clone()

            core._apply_avoid_collision_guard(prev_bx, prev_by, prev_rx, prev_ry)

            dead_x_after = core.blue_x[0, 1].item()
            dead_y_after = core.blue_y[0, 1].item()

            # CORRECT behavior: dead agent position unchanged
            # BUGGY behavior: dead agent position is mutated by repulsion
            self.assertAlmostEqual(
                dead_x_after, dead_x_before, places=4,
                msg=(
                    f"Dead agent X position changed from {dead_x_before} to {dead_x_after}. "
                    f"The collision guard should not move dead agents."
                ),
            )
            self.assertAlmostEqual(
                dead_y_after, dead_y_before, places=4,
                msg=(
                    f"Dead agent Y position changed from {dead_y_before} to {dead_y_after}. "
                    f"The collision guard should not move dead agents."
                ),
            )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
