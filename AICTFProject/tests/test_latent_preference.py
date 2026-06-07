from __future__ import annotations

import unittest
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F

from rl.custom_ppo.csv_writers import _update_fieldnames
from rl.custom_ppo.latent_strategy_state import (
    LatentStrategyState,
    _advantage_weighted_target_from_records,
    _episode_bucket_baseline_keys,
    _role_phase_specialist_context_keys,
    _router_specialist_loss,
    _specialist_context_keys_for_mode,
    _tactical_local_context_keys,
    _tactical_specialist_context_keys,
    _warmup_ramp_coef_scale,
)
from rl.ppo_core import compute_gae
from tests.test_latent_episode_warmup import _make_trainer


class LatentPreferenceTests(unittest.TestCase):
    def test_tactical_specialist_telemetry_fields_are_exposed(self) -> None:
        fields = _update_fieldnames(use_latent_strategy=True, latent_k=4)
        for field in (
            "latent_specialist_active_buckets",
            "latent_specialist_context_bucket_entropy",
            "latent_specialist_marginal_entropy",
            "latent_specialist_loss",
            "latent_actor_z_separation_loss",
            "latent_actor_z_separation_jsd",
            "latent_tactical_bucket_fallback_fraction",
            "bucket_baseline_count",
            "z_change_count",
            "z_dwell_mean",
            "z_refresh_attempt_count",
            "z_refresh_accept_count",
            "z_refresh_reject_dwell_count",
            "z_refresh_reason_interval",
            "z_refresh_reason_flag",
            "z_refresh_reason_phase",
            "z_refresh_reason_score_pressure",
            "q_phi_argmax_vs_executed_z_agreement",
            "MI_executed_z_phase",
            "MI_executed_z_flag",
            "MI_executed_z_outcome",
        ):
            with self.subTest(field=field):
                self.assertIn(field, fields)

    @staticmethod
    def _sparse_refresh_trainer(
        *,
        warmup: int = 5,
        interval: int = 32,
        min_dwell: int = 16,
    ) -> SimpleNamespace:
        trainer = _make_trainer(
            n_envs=1,
            warmup=warmup,
            episode_credit=True,
            gs_dim=34,
        )
        trainer.latent_event_refresh_enabled = False
        trainer.latent_sparse_tactical_refresh_enabled = True
        trainer.latent_sparse_tactical_refresh_interval_steps = interval
        trainer.latent_sparse_tactical_refresh_min_dwell_steps = min_dwell
        return trainer

    @staticmethod
    def _neutral_tactical_state(z_signal: int = 0) -> torch.Tensor:
        state = torch.zeros((1, 34), dtype=torch.float32)
        state[:, 0] = float(z_signal)
        state[:, 8] = 0.8
        state[:, 9] = 0.8
        state[:, 19] = 0.5
        state[:, 20] = 0.5
        return state

    def _armed_sparse_state(
        self,
        *,
        current_z: int = 0,
        min_dwell: int = 16,
        interval: int = 32,
    ) -> tuple[LatentStrategyState, torch.Tensor]:
        trainer = self._sparse_refresh_trainer(
            warmup=5,
            interval=interval,
            min_dwell=min_dwell,
        )
        latent_state = LatentStrategyState(trainer)
        latent_state.reset()
        latent_state.current_z.fill_(current_z)
        latent_state.needs_strategy_sample.zero_()
        latent_state.episode_strategy_committed.fill_(True)
        latent_state.steps_since_z_change.fill_(min_dwell)
        latent_state.steps_since_last_tactical_refresh.zero_()
        base = self._neutral_tactical_state(current_z)
        latent_state.prev_global_state = base.clone()
        return latent_state, base

    def test_sparse_refresh_does_not_run_before_warmup_commit(self) -> None:
        trainer = self._sparse_refresh_trainer(warmup=5, min_dwell=1)
        latent_state = LatentStrategyState(trainer)
        latent_state.reset()

        state = self._neutral_tactical_state(1)
        latent_state.strategy_for_step(state)
        latent_state.mark_strategy_step_done(np.array([False]))

        transitioned = self._neutral_tactical_state(2)
        transitioned[:, 10] = 1.0
        z_idx, _, aux = latent_state.strategy_for_step(transitioned)

        self.assertEqual(int(z_idx.item()), 1)
        self.assertFalse(bool(aux["z_persist_mask"].item()))
        stats = latent_state.sparse_tactical_refresh_rollout_stats()
        self.assertEqual(stats["z_refresh_attempt_count"], 0.0)
        self.assertEqual(stats["z_refresh_accept_count"], 0.0)

    def test_sparse_refresh_rejects_transition_before_min_dwell(self) -> None:
        latent_state, _ = self._armed_sparse_state(min_dwell=16)
        latent_state.steps_since_z_change.fill_(15)
        transitioned = self._neutral_tactical_state(1)
        transitioned[:, 10] = 1.0

        z_idx, _, aux = latent_state.strategy_for_step(transitioned)

        self.assertEqual(int(z_idx.item()), 0)
        self.assertFalse(bool(aux["z_persist_mask"].item()))
        stats = latent_state.sparse_tactical_refresh_rollout_stats()
        self.assertEqual(stats["z_refresh_attempt_count"], 1.0)
        self.assertEqual(stats["z_refresh_accept_count"], 0.0)
        self.assertEqual(stats["z_refresh_reject_dwell_count"], 1.0)

    def test_sparse_refresh_accepts_interval_after_dwell(self) -> None:
        latent_state, base = self._armed_sparse_state(min_dwell=16, interval=32)
        latent_state.steps_since_last_tactical_refresh.fill_(32)
        proposal = base.clone()
        proposal[:, 0] = 1.0

        z_idx, _, aux = latent_state.strategy_for_step(proposal)

        self.assertEqual(int(z_idx.item()), 1)
        self.assertTrue(bool(aux["z_persist_mask"].item()))
        stats = latent_state.sparse_tactical_refresh_rollout_stats()
        self.assertEqual(stats["z_refresh_reason_interval"], 1.0)
        self.assertEqual(stats["z_refresh_accept_count"], 1.0)
        self.assertEqual(stats["z_change_count"], 1.0)

    def test_sparse_refresh_accepts_tactical_transitions_after_dwell(self) -> None:
        transition_updates = {
            "flag": lambda state: state.__setitem__(
                (slice(None), 10), 1.0
            ),
            "phase": lambda state: (
                state.__setitem__((slice(None), 19), 0.1),
                state.__setitem__((slice(None), 20), 0.9),
            ),
            "score_pressure": lambda state: state.__setitem__(
                (slice(None), 16), 0.5
            ),
        }
        reason_fields = {
            "flag": "z_refresh_reason_flag",
            "phase": "z_refresh_reason_phase",
            "score_pressure": "z_refresh_reason_score_pressure",
        }
        for name, update in transition_updates.items():
            with self.subTest(reason=name):
                latent_state, base = self._armed_sparse_state()
                proposal = base.clone()
                proposal[:, 0] = 1.0
                update(proposal)

                z_idx, _, _ = latent_state.strategy_for_step(proposal)

                self.assertEqual(int(z_idx.item()), 1)
                stats = latent_state.sparse_tactical_refresh_rollout_stats()
                self.assertEqual(stats["z_refresh_accept_count"], 1.0)
                self.assertEqual(stats[reason_fields[name]], 1.0)

    def test_same_z_sparse_proposal_is_not_a_switch(self) -> None:
        latent_state, base = self._armed_sparse_state(current_z=1)
        proposal = base.clone()
        proposal[:, 10] = 1.0

        z_idx, _, aux = latent_state.strategy_for_step(proposal)

        self.assertEqual(int(z_idx.item()), 1)
        self.assertFalse(bool(aux["z_persist_mask"].item()))
        stats = latent_state.sparse_tactical_refresh_rollout_stats()
        self.assertEqual(stats["z_refresh_accept_count"], 1.0)
        self.assertEqual(stats["z_change_count"], 0.0)

    def test_accepted_sparse_switch_sets_persistence_and_gae_boundary(self) -> None:
        latent_state, base = self._armed_sparse_state(current_z=0)
        proposal = base.clone()
        proposal[:, 0] = 1.0
        proposal[:, 16] = 0.5

        z_idx, prev_z, aux = latent_state.strategy_for_step(proposal)

        self.assertEqual(int(prev_z.item()), 0)
        self.assertEqual(int(z_idx.item()), 1)
        self.assertTrue(bool(aux["z_persist_mask"].item()))
        stats = latent_state.sparse_tactical_refresh_rollout_stats()
        self.assertEqual(stats["z_change_count"], 1.0)
        self.assertEqual(stats["z_dwell_mean"], 16.0)

        rewards = torch.zeros((2, 1), dtype=torch.float32)
        values = torch.ones((2, 1), dtype=torch.float32)
        next_values = torch.ones((2, 1), dtype=torch.float32)
        terminated = torch.zeros((2, 1), dtype=torch.bool)
        latent_z = torch.stack((prev_z, z_idx))
        adv_reset, _ = compute_gae(
            rewards,
            values,
            next_values,
            terminated,
            gamma=0.9,
            gae_lambda=0.95,
            latent_z=latent_z,
            reset_gae_on_z_change=True,
        )
        adv_cont, _ = compute_gae(
            rewards,
            values,
            next_values,
            terminated,
            gamma=0.9,
            gae_lambda=0.95,
            latent_z=latent_z,
            reset_gae_on_z_change=False,
        )
        self.assertNotAlmostEqual(
            float(adv_reset[0, 0]),
            float(adv_cont[0, 0]),
        )

    def test_router_specialist_loss_prefers_global_balance_with_local_decisions(self) -> None:
        context_keys = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
        uniform_logits = torch.zeros((8, 4), dtype=torch.float32)
        specialist_logits = torch.tensor(
            [
                [4.0, 0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0, 0.0],
                [0.0, 4.0, 0.0, 0.0],
                [0.0, 4.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0],
                [0.0, 0.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 4.0],
                [0.0, 0.0, 0.0, 4.0],
            ],
            dtype=torch.float32,
        )

        uniform_loss, uniform_stats = _router_specialist_loss(
            uniform_logits,
            context_keys=context_keys,
            latent_k=4,
            marginal_balance_coef=0.02,
            conditional_entropy_min_coef=0.015,
            context_mi_coef=0.04,
            coef_scale=1.0,
            min_bucket_count=2,
        )
        specialist_loss, specialist_stats = _router_specialist_loss(
            specialist_logits,
            context_keys=context_keys,
            latent_k=4,
            marginal_balance_coef=0.02,
            conditional_entropy_min_coef=0.015,
            context_mi_coef=0.04,
            coef_scale=1.0,
            min_bucket_count=2,
        )

        self.assertLess(float(specialist_loss.item()), float(uniform_loss.item()))
        self.assertAlmostEqual(
            float(specialist_stats["latent_specialist_marginal_entropy"].item()),
            float(uniform_stats["latent_specialist_marginal_entropy"].item()),
            places=4,
        )
        self.assertLess(
            float(specialist_stats["latent_specialist_conditional_entropy"].item()),
            float(uniform_stats["latent_specialist_conditional_entropy"].item()),
        )
        self.assertGreater(float(specialist_stats["latent_specialist_context_mi"].item()), 0.5)
        self.assertEqual(float(specialist_stats["latent_specialist_active_buckets"].item()), 4.0)

    def test_role_phase_specialist_context_keys_separate_flag_situations(self) -> None:
        states = torch.zeros((4, 34), dtype=torch.float32)
        states[0, 8] = 0.8
        states[0, 9] = 0.8
        states[1, 10] = 1.0
        states[1, 23] = 0.1
        states[2, 11] = 1.0
        states[2, 8] = 0.1
        states[2, 23] = 0.8
        states[3, 9] = 0.1

        keys = _role_phase_specialist_context_keys(states, include_progress=True)

        self.assertEqual(int(keys.unique().numel()), 4)
        self.assertNotEqual(int(keys[1].item()), int(keys[2].item()))
        self.assertNotEqual(int(keys[0].item()), int(keys[3].item()))

    def test_role_phase_opponent_context_keeps_phase_primary(self) -> None:
        states = torch.zeros((3, 34), dtype=torch.float32)
        states[:, 10] = 1.0
        opponent_ids = torch.tensor([3, 5, 3], dtype=torch.long)

        phase_keys = _specialist_context_keys_for_mode(
            mode="role_phase_progress",
            states=states,
            opponent_ids=opponent_ids,
            bucket_ids=None,
        )
        hierarchical_keys = _specialist_context_keys_for_mode(
            mode="role_phase_progress_opponent",
            states=states,
            opponent_ids=opponent_ids,
            bucket_ids=None,
        )

        self.assertIsNotNone(phase_keys)
        self.assertIsNotNone(hierarchical_keys)
        self.assertEqual(int(phase_keys[0].item()), int(phase_keys[1].item()))
        self.assertNotEqual(
            int(hierarchical_keys[0].item()),
            int(hierarchical_keys[1].item()),
        )
        self.assertEqual(
            int(hierarchical_keys[0].item()) // 16,
            int(hierarchical_keys[1].item()) // 16,
        )

    def test_tactical_context_keys_cover_phase_flags_score_and_opponent(self) -> None:
        states = torch.zeros((5, 34), dtype=torch.float32)
        states[:, 8] = 0.8
        states[:, 9] = 0.8
        states[1, 10] = 1.0
        states[2, 11] = 1.0
        states[3, 16] = -0.5
        states[4, 16] = 0.5
        opponent_ids = torch.tensor([3, 3, 3, 3, 5], dtype=torch.long)

        keys = _tactical_specialist_context_keys(
            states,
            opponent_ids=opponent_ids,
        )

        self.assertEqual(int(keys.unique().numel()), 5)
        same_state_other_opponent = _tactical_specialist_context_keys(
            states[[0, 0]],
            opponent_ids=torch.tensor([3, 5], dtype=torch.long),
        )
        self.assertEqual(
            int(same_state_other_opponent[0].item()) // 16,
            int(same_state_other_opponent[1].item()) // 16,
        )
        self.assertNotEqual(
            int(same_state_other_opponent[0].item()),
            int(same_state_other_opponent[1].item()),
        )

    def test_tactical_context_keys_separate_attack_and_defense_pressure(self) -> None:
        states = torch.zeros((3, 34), dtype=torch.float32)
        states[:, 8] = 0.8
        states[:, 9] = 0.8
        states[1, 19] = 0.1
        states[1, 20] = 0.8
        states[2, 19] = 0.8
        states[2, 20] = 0.1

        keys = _tactical_local_context_keys(states)

        self.assertEqual(int(keys.unique().numel()), 3)

    def test_tactical_bucket_baseline_uses_episode_trajectory_bucket(self) -> None:
        states = torch.zeros((3, 34), dtype=torch.float32)
        states[1, 10] = 1.0
        states[2, 16] = 0.5
        opponent_ids = torch.tensor([3, 5, 6], dtype=torch.long)
        bucket_ids = torch.tensor([1, 2, 3], dtype=torch.long)

        baseline_keys = _episode_bucket_baseline_keys(
            mode="tactical_context_opponent",
            states=states,
            opponent_ids=opponent_ids,
            bucket_ids=bucket_ids,
        )

        expected = bucket_ids * 16 + opponent_ids
        self.assertTrue(torch.equal(expected, baseline_keys))

    def test_episode_tactical_bucket_prefers_meaningful_trajectory_state(self) -> None:
        trainer = _make_trainer(
            n_envs=1,
            warmup=0,
            episode_credit=True,
            gs_dim=34,
        )
        latent_state = LatentStrategyState(trainer)
        latent_state.reset()
        neutral = torch.zeros((1, 34), dtype=torch.float32)
        neutral[:, 8] = 0.8
        neutral[:, 9] = 0.8
        attack = neutral.clone()
        attack[:, 19] = 0.1
        attack[:, 20] = 0.8

        for _ in range(5):
            latent_state.record_tactical_context_step(neutral)
        for _ in range(2):
            latent_state.record_tactical_context_step(attack)

        attack_bucket = int(_tactical_local_context_keys(attack)[0].item())
        self.assertEqual(
            latent_state.representative_tactical_bucket(0),
            attack_bucket,
        )

    def test_missing_trajectory_preserves_contrast_bucket_and_logs_fallback(
        self,
    ) -> None:
        trainer = _make_trainer(
            n_envs=1,
            warmup=0,
            episode_credit=True,
            gs_dim=34,
        )
        trainer.cfg = SimpleNamespace(
            opponent_pool=["OP3"],
            opponent_pool_weights=[1.0],
        )
        latent_state = LatentStrategyState(trainer)
        latent_state.reset()
        latent_state.episode_forced_z[0] = True
        latent_state.episode_forced_z_id[0] = 2
        latent_state.episode_contrast_bucket[0] = 5
        latent_state.episode_behavior_count[0] = 1

        latent_state.record_episode_strategy_outcome(
            0,
            {"scripted_tag": "OP3"},
            episode_return=1.0,
        )

        self.assertEqual(
            latent_state.latent_preference_buffer[-1]["context_bucket"],
            5,
        )
        stats = latent_state.behavior_contrast_rollout_stats()
        self.assertEqual(
            stats["latent_tactical_bucket_fallback_fraction"],
            1.0,
        )

    def test_missing_trajectory_without_legacy_bucket_uses_nonzero_neutral(
        self,
    ) -> None:
        trainer = _make_trainer(
            n_envs=1,
            warmup=0,
            episode_credit=True,
            gs_dim=34,
        )
        latent_state = LatentStrategyState(trainer)
        latent_state.reset()

        self.assertEqual(latent_state.representative_tactical_bucket(0), 1)

    def test_rollout_specialist_router_uses_tactical_states_after_warmup(self) -> None:
        class RouterModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.global_state_dim = 34
                self.strategy_encoder = torch.nn.Linear(34, 4)

            def strategy_logits(self, state: torch.Tensor) -> torch.Tensor:
                return self.strategy_encoder(state)

        model = RouterModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = SimpleNamespace(
            env=SimpleNamespace(num_envs=1),
            device=torch.device("cpu"),
            model=model,
            optimizer=optimizer,
            latent_router_optimizer=optimizer,
            cfg=SimpleNamespace(max_grad_norm=1.0),
            use_latent_strategy=True,
            fixed_latent_strategy=False,
            fixed_latent_strategy_id=0,
            latent_k=4,
            latent_kl_consecutive=0.0,
            temporal_tracker=None,
            _last_context_state=None,
            latent_specialist_router_enabled=True,
            latent_specialist_use_rollout_states=True,
            latent_specialist_rollout_max_samples=64,
            latent_specialist_context_key_mode=(
                "tactical_phase_flags_score_opponent"
            ),
            latent_specialist_warmup_steps=100,
            latent_specialist_ramp_steps=100,
            latent_specialist_min_bucket_count=2,
            latent_specialist_conditional_entropy_scope="context_bucket",
            latent_conditional_entropy_min_coef_start=0.01,
            latent_conditional_entropy_min_coef=0.05,
            latent_marginal_balance_coef=0.02,
            latent_context_mi_coef=0.05,
            latent_resample_every_n=0,
            latent_episode_strategy_ppo=True,
            latent_episode_strategy_warmup_decision_steps=0,
        )
        latent_state = LatentStrategyState(trainer)

        states = torch.zeros((12, 1, 34), dtype=torch.float32)
        states[:, :, 8] = 0.8
        states[:, :, 9] = 0.8
        states[3:6, :, 19] = 0.1
        states[3:6, :, 20] = 0.8
        states[6:9, :, 19] = 0.8
        states[6:9, :, 20] = 0.1
        states[9:12, :, 11] = 1.0
        opponent_ids = torch.tensor(
            [2, 4, 5, 2, 4, 5, 2, 4, 5, 2, 4, 5],
            dtype=torch.long,
        ).reshape(12, 1)
        states = states.repeat_interleave(2, dim=0)
        opponent_ids = opponent_ids.repeat_interleave(2, dim=0)
        buffer = SimpleNamespace(
            pos=24,
            fields={
                "global_state": states,
                "opponent_id": opponent_ids,
            },
        )

        trainer.global_step = 99
        before = model.strategy_encoder.weight.detach().clone()
        cold_stats = latent_state.apply_rollout_specialist_router(buffer)
        self.assertEqual(cold_stats["latent_specialist_coef_scale"], 0.0)
        self.assertTrue(
            torch.equal(before, model.strategy_encoder.weight.detach())
        )

        trainer.global_step = 200
        hot_stats = latent_state.apply_rollout_specialist_router(buffer)
        self.assertEqual(hot_stats["latent_specialist_rollout_samples"], 24.0)
        self.assertGreater(hot_stats["latent_specialist_active_buckets"], 3.0)
        self.assertFalse(
            torch.equal(before, model.strategy_encoder.weight.detach())
        )

    def test_bucket_conditional_entropy_prefers_coherent_local_niches(self) -> None:
        context_keys = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        coherent_logits = torch.tensor(
            [
                [6.0, 0.0, 0.0, 0.0],
                [6.0, 0.0, 0.0, 0.0],
                [0.0, 6.0, 0.0, 0.0],
                [0.0, 6.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        incoherent_logits = torch.tensor(
            [
                [6.0, 0.0, 0.0, 0.0],
                [0.0, 6.0, 0.0, 0.0],
                [6.0, 0.0, 0.0, 0.0],
                [0.0, 6.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

        coherent_loss, coherent_stats = _router_specialist_loss(
            coherent_logits,
            context_keys=context_keys,
            latent_k=4,
            marginal_balance_coef=0.02,
            conditional_entropy_min_coef=0.05,
            conditional_entropy_min_coef_start=0.01,
            conditional_entropy_scope="context_bucket",
            context_mi_coef=0.0,
            coef_scale=1.0,
            min_bucket_count=2,
        )
        incoherent_loss, incoherent_stats = _router_specialist_loss(
            incoherent_logits,
            context_keys=context_keys,
            latent_k=4,
            marginal_balance_coef=0.02,
            conditional_entropy_min_coef=0.05,
            conditional_entropy_min_coef_start=0.01,
            conditional_entropy_scope="context_bucket",
            context_mi_coef=0.0,
            coef_scale=1.0,
            min_bucket_count=2,
        )

        self.assertLess(float(coherent_loss.item()), float(incoherent_loss.item()))
        self.assertLess(
            float(
                coherent_stats[
                    "latent_specialist_context_bucket_entropy"
                ].item()
            ),
            float(
                incoherent_stats[
                    "latent_specialist_context_bucket_entropy"
                ].item()
            ),
        )
        self.assertAlmostEqual(
            float(
                coherent_stats["latent_specialist_conditional_coef"].item()
            ),
            0.05,
        )

    def test_advantage_weighted_target_requires_clear_margin(self) -> None:
        weak_records = [
            {"z": 0, "win_loss": 1},
            {"z": 0, "win_loss": 0},
            {"z": 1, "win_loss": 1},
            {"z": 1, "win_loss": 0},
        ]
        target, stats = _advantage_weighted_target_from_records(
            weak_records,
            latent_k=4,
            min_count=4,
            min_distinct_z=2,
            temperature=0.35,
            margin_threshold=0.15,
        )
        self.assertIsNone(target)
        self.assertAlmostEqual(stats["margin"], 0.0)

        strong_records = [
            {"z": 2, "win_loss": 1},
            {"z": 2, "win_loss": 1},
            {"z": 3, "win_loss": 0},
            {"z": 3, "win_loss": 1},
        ]
        target, stats = _advantage_weighted_target_from_records(
            strong_records,
            latent_k=4,
            min_count=4,
            min_distinct_z=2,
            temperature=0.35,
            margin_threshold=0.15,
        )
        self.assertIsNotNone(target)
        self.assertEqual(int(stats["best_z"]), 2)
        self.assertGreater(stats["margin"], 0.15)
        self.assertGreater(float(target[2]), float(target[3]))

    def test_awrd_warmup_ramp_schedule(self) -> None:
        self.assertEqual(
            _warmup_ramp_coef_scale(
                global_step=99_999,
                warmup_steps=100_000,
                ramp_steps=300_000,
            ),
            0.0,
        )
        self.assertAlmostEqual(
            _warmup_ramp_coef_scale(
                global_step=250_000,
                warmup_steps=100_000,
                ramp_steps=300_000,
            ),
            0.5,
        )
        self.assertEqual(
            _warmup_ramp_coef_scale(
                global_step=400_000,
                warmup_steps=100_000,
                ramp_steps=300_000,
            ),
            1.0,
        )

    def test_record_episode_strategy_outcome_forced_z(self) -> None:
        trainer = _make_trainer(n_envs=2, warmup=0, episode_credit=True, gs_dim=4)
        trainer.cfg = SimpleNamespace(opponent_pool=["OP3", "OP5"], opponent_pool_weights=[0.5, 0.5])
        trainer.latent_preference_coef = 0.03
        trainer.episode_stats = SimpleNamespace(episodes_completed=0)
        
        latent_state = LatentStrategyState(trainer)
        latent_state.reset()
        
        # Mark env 0 as forced-z episode
        latent_state.episode_forced_z[0] = True
        latent_state.episode_forced_z_id[0] = 2
        latent_state.episode_contrast_bucket[0] = 5
        latent_state.episode_behavior_sum[0] = torch.ones(13, dtype=torch.float32)
        latent_state.episode_behavior_count[0] = 1
        
        # Mark env 1 as regular (non-forced-z) episode with a started strategy
        latent_state.episode_forced_z[1] = False
        latent_state.episode_strategy_has_start[1] = True
        latent_state.episode_strategy_z[1] = 1
        latent_state.episode_strategy_log_prob[1] = -0.5
        latent_state.episode_strategy_bucket[1] = 6
        
        # Record outcome for env 0 (forced-z)
        info_forced = {"scripted_tag": "OP3"}
        latent_state.record_episode_strategy_outcome(0, info_forced, episode_return=5.5)
        
        # Should record in latent_preference_buffer
        self.assertEqual(len(latent_state.latent_preference_buffer), 1)
        record = latent_state.latent_preference_buffer[0]
        self.assertEqual(record["context_bucket"], 5)
        self.assertEqual(record["opponent"], 2)  # OP3 maps to index 2
        self.assertEqual(record["phase_flag_state"], 5)
        self.assertEqual(record["z"], 2)
        self.assertAlmostEqual(record["return"], 5.5)
        
        # Standard rollout records should be empty
        self.assertEqual(len(latent_state.rollout_strategy_episode_records), 0)

        # Record outcome for env 1 (regular)
        info_reg = {"scripted_tag": "OP5"}
        latent_state.record_episode_strategy_outcome(1, info_reg, episode_return=2.0)
        
        # Standard rollout records should now have 1 item
        self.assertEqual(len(latent_state.rollout_strategy_episode_records), 1)
        # Latent preference buffer should still have only 1 item (the forced-z one)
        self.assertEqual(len(latent_state.latent_preference_buffer), 1)

    def test_apply_episode_strategy_ppo_pref_loss(self) -> None:
        trainer = _make_trainer(n_envs=1, warmup=0, episode_credit=True, gs_dim=4)
        trainer.cfg = SimpleNamespace(opponent_pool=["OP3", "OP5"], opponent_pool_weights=[0.5, 0.5])
        trainer.latent_preference_coef = 0.03
        trainer.latent_preference_temperature = 1.0
        trainer.latent_preference_min_bucket_count = 3
        trainer.latent_preference_min_distinct_z = 2
        
        class MockOptimizer:
            def __init__(self):
                self.zero_grad_called = False
                self.step_called = False
                self.param_groups = [{"params": []}]
            def zero_grad(self, set_to_none=True):
                self.zero_grad_called = True
            def step(self):
                self.step_called = True
        
        trainer.optimizer = MockOptimizer()
        trainer.latent_router_optimizer = None
        trainer.cfg.max_grad_norm = 1.0
        trainer.latent_episode_strategy_coef = 0.3
        trainer.latent_episode_strategy_value_coef = 0.5
        trainer.latent_entropy_objective = "maximize"
        trainer.latent_episode_strategy_n_epochs = 1
        trainer.latent_episode_strategy_clip_eps = 0.2
        trainer.latent_episode_strategy_return_norm = False
        trainer.latent_lam_h = 0.0
        
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.global_state_dim = 4
                self.strategy_encoder = torch.nn.Linear(4, 4)
                self.episode_strategy_value_head = torch.nn.Linear(4, 1)
            def strategy_logits(self, state):
                return self.strategy_encoder(state)
            def episode_strategy_value(self, state, z):
                return self.episode_strategy_value_head(state).squeeze(-1)
        
        mock_model = MockModel()
        trainer.model = mock_model
        
        latent_state = LatentStrategyState(trainer)
        latent_state.reset()
        
        # Add 3 records to the preference buffer for the same bucket key:
        # Opponent id = 2 (OP3), context_bucket = 5. Key = 2 * 256 + 5 = 517.
        # This satisfies min_bucket_count=3 and min_distinct_z=2 (z=0, z=1).
        latent_state.latent_preference_buffer.append({
            "context_bucket": 5,
            "opponent": 2,
            "phase_flag_state": 5,
            "z": 0,
            "return": 10.0,
            "behavior_embedding": [0.0]*13,
            "win_loss": 0,
        })
        latent_state.latent_preference_buffer.append({
            "context_bucket": 5,
            "opponent": 2,
            "phase_flag_state": 5,
            "z": 1,
            "return": 20.0,
            "behavior_embedding": [0.0]*13,
            "win_loss": 0,
        })
        latent_state.latent_preference_buffer.append({
            "context_bucket": 5,
            "opponent": 2,
            "phase_flag_state": 5,
            "z": 1,
            "return": 30.0,
            "behavior_embedding": [0.0]*13,
            "win_loss": 0,
        })
        
        # Put 1 matching episode record in standard training rollout records
        latent_state.rollout_strategy_episode_records.append({
            "episode_id": 0,
            "global_state_0": torch.zeros(4, dtype=torch.float32),
            "z": 1,
            "z_logprob_old": 0.0,
            "episode_return": 15.0,
            "bucket_id": 5,
            "opponent_id": 2,
            "q_phi_probs": [0.25]*4,
        })
        
        stats = latent_state.apply_episode_strategy_ppo(latent_lam_h=0.0)
        
        # Verify stats logged from preference update
        self.assertGreater(stats["latent_preference_loss"], 0.0)
        self.assertEqual(stats["latent_preference_active_fraction"], 1.0)
        self.assertEqual(stats["latent_preference_buffer_size"], 3)
        self.assertEqual(stats["latent_preference_num_active_buckets"], 1)
        self.assertGreater(stats["latent_preference_target_entropy"], 0.0)
        self.assertTrue(trainer.optimizer.zero_grad_called)
        self.assertTrue(trainer.optimizer.step_called)

    def test_apply_episode_strategy_ppo_opponent_balanced_telemetry(self) -> None:
        trainer = _make_trainer(n_envs=3, warmup=0, episode_credit=True, gs_dim=4)
        trainer.cfg = SimpleNamespace(opponent_pool=["OP3", "OP5", "OP6"], opponent_pool_weights=[0.33, 0.33, 0.34])
        trainer.latent_preference_coef = 0.03
        trainer.latent_preference_temperature = 1.0
        trainer.latent_preference_min_bucket_count = 3
        trainer.latent_preference_min_distinct_z = 2
        
        # Turn on opponent balanced loss and telemetry logging
        trainer.cfg.latent_preference_opponent_balanced = True
        trainer.cfg.latent_preference_log_opponent_targets = True
        
        class MockOptimizer:
            def __init__(self):
                self.zero_grad_called = False
                self.step_called = False
                self.param_groups = [{"params": []}]
            def zero_grad(self, set_to_none=True):
                self.zero_grad_called = True
            def step(self):
                self.step_called = True
        
        trainer.optimizer = MockOptimizer()
        trainer.latent_router_optimizer = None
        trainer.cfg.max_grad_norm = 1.0
        trainer.latent_episode_strategy_coef = 0.3
        trainer.latent_episode_strategy_value_coef = 0.5
        trainer.latent_entropy_objective = "maximize"
        trainer.latent_episode_strategy_n_epochs = 1
        trainer.latent_episode_strategy_clip_eps = 0.2
        trainer.latent_episode_strategy_return_norm = False
        trainer.latent_lam_h = 0.0
        
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.global_state_dim = 4
                self.strategy_encoder = torch.nn.Linear(4, 4)
                self.episode_strategy_value_head = torch.nn.Linear(4, 1)
                
                # Setup dummy weights to yield deterministic predictions
                with torch.no_grad():
                    self.strategy_encoder.weight.zero_()
                    self.strategy_encoder.bias.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
            def strategy_logits(self, state):
                return self.strategy_encoder(state)
            def episode_strategy_value(self, state, z):
                return self.episode_strategy_value_head(state).squeeze(-1)
        
        mock_model = MockModel()
        trainer.model = mock_model
        
        latent_state = LatentStrategyState(trainer)
        latent_state.reset()
        
        # Add 3 records to the preference buffer for OP5 (id 4), bucket 5
        latent_state.latent_preference_buffer.append({
            "context_bucket": 5, "opponent": 4, "phase_flag_state": 5, "z": 0, "return": 10.0, "behavior_embedding": [0.0]*13, "win_loss": 0,
        })
        latent_state.latent_preference_buffer.append({
            "context_bucket": 5, "opponent": 4, "phase_flag_state": 5, "z": 1, "return": 20.0, "behavior_embedding": [0.0]*13, "win_loss": 0,
        })
        latent_state.latent_preference_buffer.append({
            "context_bucket": 5, "opponent": 4, "phase_flag_state": 5, "z": 1, "return": 30.0, "behavior_embedding": [0.0]*13, "win_loss": 0,
        })
        
        # Add 3 records to the preference buffer for OP6 (id 5), bucket 6
        latent_state.latent_preference_buffer.append({
            "context_bucket": 6, "opponent": 5, "phase_flag_state": 6, "z": 2, "return": 40.0, "behavior_embedding": [0.0]*13, "win_loss": 0,
        })
        latent_state.latent_preference_buffer.append({
            "context_bucket": 6, "opponent": 5, "phase_flag_state": 6, "z": 3, "return": 50.0, "behavior_embedding": [0.0]*13, "win_loss": 0,
        })
        latent_state.latent_preference_buffer.append({
            "context_bucket": 6, "opponent": 5, "phase_flag_state": 6, "z": 3, "return": 60.0, "behavior_embedding": [0.0]*13, "win_loss": 0,
        })
        
        # Add 3 episodes to standard rollout records:
        # Two for OP5 (id 4), one for OP6 (id 5)
        latent_state.rollout_strategy_episode_records.append({
            "episode_id": 0, "global_state_0": torch.zeros(4, dtype=torch.float32), "z": 1, "z_logprob_old": 0.0, "episode_return": 15.0, "bucket_id": 5, "opponent_id": 4, "q_phi_probs": [0.25]*4,
        })
        latent_state.rollout_strategy_episode_records.append({
            "episode_id": 1, "global_state_0": torch.zeros(4, dtype=torch.float32), "z": 1, "z_logprob_old": 0.0, "episode_return": 25.0, "bucket_id": 5, "opponent_id": 4, "q_phi_probs": [0.25]*4,
        })
        latent_state.rollout_strategy_episode_records.append({
            "episode_id": 2, "global_state_0": torch.zeros(4, dtype=torch.float32), "z": 3, "z_logprob_old": 0.0, "episode_return": 55.0, "bucket_id": 6, "opponent_id": 5, "q_phi_probs": [0.25]*4,
        })
        
        stats = latent_state.apply_episode_strategy_ppo(latent_lam_h=0.0)
        
        # Verify specific opponent buffer count
        self.assertEqual(stats["latent_pref_op5_buffer_count"], 3.0)
        self.assertEqual(stats["latent_pref_op6_buffer_count"], 3.0)
        
        # Verify active fraction
        self.assertAlmostEqual(stats["latent_pref_op5_active_fraction"], 1.0)
        self.assertAlmostEqual(stats["latent_pref_op6_active_fraction"], 1.0)
        
        # Verify active buckets
        self.assertEqual(stats["latent_pref_op5_active_buckets"], 1.0)
        self.assertEqual(stats["latent_pref_op6_active_buckets"], 1.0)
        
        # Verify best z
        self.assertEqual(stats["latent_pref_op5_best_z"], 1.0)
        self.assertEqual(stats["latent_pref_op6_best_z"], 3.0)
        
        # Verify target entropy
        self.assertGreater(stats["latent_pref_op5_target_entropy"], 0.0)
        self.assertGreater(stats["latent_pref_op6_target_entropy"], 0.0)
        
        # Verify target distributions
        self.assertGreater(stats["latent_pref_op5_target_z1"], 0.5)
        self.assertGreater(stats["latent_pref_op6_target_z3"], 0.5)
        
        # Verify individual opponent losses
        self.assertGreater(stats["latent_pref_op5_loss"], 0.0)
        self.assertGreater(stats["latent_pref_op6_loss"], 0.0)
        
        # Since opponent_balanced = True, the overall preference loss should be the average
        # of the two individual opponent losses: (OP5_loss + OP6_loss) / 2
        expected_balanced_loss = (stats["latent_pref_op5_loss"] + stats["latent_pref_op6_loss"]) / 2.0
        self.assertAlmostEqual(stats["latent_preference_loss"], expected_balanced_loss, places=5)


    def test_v3h2_confidence_weighted_loss(self) -> None:
        trainer = _make_trainer(n_envs=1, warmup=0, episode_credit=True, gs_dim=4)
        trainer.cfg = SimpleNamespace(opponent_pool=["OP3", "OP5"], opponent_pool_weights=[0.5, 0.5])
        
        # v3h2 hyperparams
        trainer.latent_preference_coef = 0.03
        trainer.latent_preference_temperature = 1.0
        trainer.latent_preference_min_bucket_count = 2
        trainer.latent_preference_min_distinct_z = 2
        trainer.latent_preference_confidence_scale = 2.0
        trainer.latent_preference_commit_coef = 0.003
        trainer.late_entropy_floor = 0.0003
        trainer.commitment_type = "confidence_weighted_entropy"
        
        class MockOptimizer:
            def __init__(self):
                self.zero_grad_called = False
                self.step_called = False
                self.param_groups = [{"params": []}]
            def zero_grad(self, set_to_none=True):
                self.zero_grad_called = True
            def step(self):
                self.step_called = True
        
        trainer.optimizer = MockOptimizer()
        trainer.latent_router_optimizer = None
        trainer.cfg.max_grad_norm = 1.0
        trainer.latent_episode_strategy_coef = 0.3
        trainer.latent_episode_strategy_value_coef = 0.5
        trainer.latent_entropy_objective = "maximize"
        trainer.latent_episode_strategy_n_epochs = 1
        trainer.latent_episode_strategy_clip_eps = 0.2
        trainer.latent_episode_strategy_return_norm = False
        trainer.latent_lam_h = 0.0
        
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.global_state_dim = 4
                self.strategy_encoder = torch.nn.Linear(4, 4)
                self.episode_strategy_value_head = torch.nn.Linear(4, 1)
                
                # Setup dummy weights to yield deterministic predictions
                with torch.no_grad():
                    self.strategy_encoder.weight.zero_()
                    self.strategy_encoder.bias.copy_(torch.tensor([0.0, 0.0, 0.0, 0.0]))
            def strategy_logits(self, state):
                return self.strategy_encoder(state)
            def episode_strategy_value(self, state, z):
                return self.episode_strategy_value_head(state).squeeze(-1)
        
        mock_model = MockModel()
        trainer.model = mock_model
        
        latent_state = LatentStrategyState(trainer)
        latent_state.reset()
        
        # Add 2 records to the preference buffer:
        latent_state.latent_preference_buffer.append({
            "context_bucket": 5, "opponent": 2, "phase_flag_state": 5, "z": 0, "return": 0.0, "behavior_embedding": [0.0]*13, "win_loss": 0,
        })
        latent_state.latent_preference_buffer.append({
            "context_bucket": 5, "opponent": 2, "phase_flag_state": 5, "z": 1, "return": 100.0, "behavior_embedding": [0.0]*13, "win_loss": 0,
        })
        
        # Put 1 matching episode record in standard training rollout records
        latent_state.rollout_strategy_episode_records.append({
            "episode_id": 0, "global_state_0": torch.zeros(4, dtype=torch.float32), "z": 1, "z_logprob_old": 0.0, "episode_return": 15.0, "bucket_id": 5, "opponent_id": 2, "q_phi_probs": [0.25]*4,
        })
        
        stats = latent_state.apply_episode_strategy_ppo(latent_lam_h=0.0)
        self.assertGreater(stats["latent_preference_loss"], 0.0)

    def test_apply_episode_strategy_ppo_v3i3_event_pref_normalization(self) -> None:
        trainer = _make_trainer(n_envs=1, warmup=0, episode_credit=True, gs_dim=4)
        trainer.cfg = SimpleNamespace(opponent_pool=["OP3"], opponent_pool_weights=[1.0])
        trainer.latent_v3i3_event_preference_enabled = True
        trainer.latent_v3i3_event_preference_coef = 0.5
        trainer.latent_v3i3_event_preference_temperature = 1.0
        trainer.latent_v3i3_event_preference_min_bucket_count = 3
        trainer.latent_v3i3_event_preference_min_distinct_z = 1
        trainer.latent_v3i3_event_preference_buffer_size = 1000
        trainer.latent_v3i3_event_preference_warmup_steps = 0
        trainer.latent_v3i3_event_preference_normalize = True
        trainer.global_step = 100
        trainer.latent_k = 4
        trainer.latent_episode_strategy_coef = 0.0
        trainer.latent_episode_strategy_n_epochs = 1
        trainer.cfg.max_grad_norm = 1.0
        trainer.latent_episode_strategy_return_norm = False
        trainer.latent_episode_strategy_value_coef = 0.5
        trainer.latent_entropy_objective = "maximize"
        trainer.latent_episode_strategy_clip_eps = 0.2
        trainer.latent_lam_h = 0.0
        trainer.latent_preference_coef = 0.0
        trainer.latent_event_preference_key_mode = "event_flag"

        class MockOptimizer:
            def __init__(self):
                self.zero_grad_called = False
                self.step_called = False
                self.param_groups = [{"params": []}]
            def zero_grad(self, set_to_none=True):
                self.zero_grad_called = True
            def step(self):
                self.step_called = True

        trainer.optimizer = MockOptimizer()
        trainer.latent_router_optimizer = None

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.global_state_dim = 4
                self.strategy_encoder = torch.nn.Linear(4, 4)
                self.episode_strategy_value_head = torch.nn.Linear(4, 1)
            def strategy_logits(self, state):
                return self.strategy_encoder(state)
            def episode_strategy_value(self, state, z):
                return self.episode_strategy_value_head(state).squeeze(-1)

        trainer.model = MockModel()

        latent_state = LatentStrategyState(trainer)
        latent_state.reset()

        # Add records to refresh_preference_buffer from two different flag states:
        # Key A: opponent=2 (OP3), event=1, flag=5. Returns: z=1 -> 20.0. Baseline A = 20.0. Normalized = 0.0.
        # Key B: opponent=2 (OP3), event=1, flag=6. Returns: z=0 -> 120.0. Baseline B = 120.0. Normalized = 0.0.
        # Min bucket count = 3, so full lookup for Key A or Key B alone fails (since counts are < 3).
        # It falls back to oe level (opponent=2, event=1), combining Key A and Key B.
        # If we normalize, both resolved means for z=0 and z=1 will be 0.0, leading to a uniform resolved target.
        latent_state.refresh_preference_buffer.append({
            "opponent_id": 2, "event_type": 1, "flag_state_bucket": 5, "z": 1, "future_return": 20.0,
        })
        latent_state.refresh_preference_buffer.append({
            "opponent_id": 2, "event_type": 1, "flag_state_bucket": 5, "z": 1, "future_return": 20.0,
        })
        latent_state.refresh_preference_buffer.append({
            "opponent_id": 2, "event_type": 1, "flag_state_bucket": 6, "z": 0, "future_return": 120.0,
        })
        latent_state.refresh_preference_buffer.append({
            "opponent_id": 2, "event_type": 1, "flag_state_bucket": 6, "z": 0, "future_return": 120.0,
        })

        # Put matching records in rollout_refresh_records
        # Target lookup at (opponent=2, reason=1, flag=5) falls back to (2, 1).
        latent_state.rollout_refresh_records.append({
            "refresh_state": torch.zeros(4, dtype=torch.float32),
            "opponent_id": 2,
            "reason_id": 1,
            "flag_state_bucket": 5,
            "next_z": 1,
            "return_at_refresh": 0.0,
        })

        # Matching episode record to prevent empty check
        latent_state.rollout_strategy_episode_records.append({
            "episode_id": 0, "global_state_0": torch.zeros(4, dtype=torch.float32), "z": 1, "z_logprob_old": 0.0, "episode_return": 15.0, "bucket_id": 5, "opponent_id": 2, "q_phi_probs": [0.25]*4,
        })

        stats = latent_state.apply_episode_strategy_ppo(latent_lam_h=0.0)

        # Telemetry counts verify active records and buckets
        self.assertEqual(stats["latent_v3i3_event_pref_buffer_size"], 4.0)
        self.assertEqual(stats["latent_v3i3_event_pref_rollout_records"], 1.0)
        self.assertEqual(stats["latent_v3i3_event_pref_active_records"], 1.0)
        self.assertEqual(stats["latent_v3i3_event_pref_active_buckets"], 1.0)
        self.assertEqual(stats["latent_v3i3_event_pref_fallback_oe"], 1.0)
        self.assertEqual(stats["latent_v3i3_event_pref_fallback_full"], 0.0)

        # Target entropy for normalized target (should be close to uniform ln 4 = 1.386)
        # Because z=0 and z=1 normalized returns are both 0.0, means=[0.0, 0.0, 0.0, 0.0] -> uniform.
        self.assertAlmostEqual(stats["latent_v3i3_event_pref_target_entropy"], 1.38629436, places=4)

    def test_apply_episode_strategy_ppo_v3i4_normalizes_by_progress_key(self) -> None:
        trainer = _make_trainer(n_envs=1, warmup=0, episode_credit=True, gs_dim=4)
        trainer.cfg = SimpleNamespace(opponent_pool=["OP3"], opponent_pool_weights=[1.0])
        trainer.latent_v3i3_event_preference_enabled = True
        trainer.latent_v3i3_event_preference_coef = 0.5
        trainer.latent_v3i3_event_preference_temperature = 1.0
        trainer.latent_v3i3_event_preference_min_bucket_count = 3
        trainer.latent_v3i3_event_preference_min_distinct_z = 1
        trainer.latent_v3i3_event_preference_buffer_size = 1000
        trainer.latent_v3i3_event_preference_warmup_steps = 0
        trainer.latent_v3i3_event_preference_normalize = True
        trainer.global_step = 100
        trainer.latent_k = 4
        trainer.latent_episode_strategy_coef = 0.0
        trainer.latent_episode_strategy_n_epochs = 1
        trainer.cfg.max_grad_norm = 1.0
        trainer.latent_episode_strategy_return_norm = False
        trainer.latent_episode_strategy_value_coef = 0.5
        trainer.latent_entropy_objective = "maximize"
        trainer.latent_episode_strategy_clip_eps = 0.2
        trainer.latent_lam_h = 0.0
        trainer.latent_preference_coef = 0.0
        trainer.latent_event_preference_key_mode = "event_flag_progress"

        class MockOptimizer:
            def __init__(self):
                self.zero_grad_called = False
                self.step_called = False
                self.param_groups = [{"params": []}]
            def zero_grad(self, set_to_none=True):
                self.zero_grad_called = True
            def step(self):
                self.step_called = True

        trainer.optimizer = MockOptimizer()
        trainer.latent_router_optimizer = None

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.global_state_dim = 4
                self.strategy_encoder = torch.nn.Linear(4, 4)
                self.episode_strategy_value_head = torch.nn.Linear(4, 1)
            def strategy_logits(self, state):
                return self.strategy_encoder(state)
            def episode_strategy_value(self, state, z):
                return self.episode_strategy_value_head(state).squeeze(-1)

        trainer.model = MockModel()

        latent_state = LatentStrategyState(trainer)
        latent_state.reset()

        # Same opponent/event/flag, different progress buckets.
        # Correct v3i4 normalization subtracts each full progress-key baseline,
        # so fallback to (opp,event,flag) sees zero advantage for both z slots.
        for _ in range(2):
            latent_state.refresh_preference_buffer.append({
                "opponent_id": 2,
                "event_type": 1,
                "flag_state_bucket": 2,
                "carrier_progress_bucket": 1,
                "z": 1,
                "future_return": 20.0,
            })
            latent_state.refresh_preference_buffer.append({
                "opponent_id": 2,
                "event_type": 1,
                "flag_state_bucket": 2,
                "carrier_progress_bucket": 3,
                "z": 0,
                "future_return": 120.0,
            })

        latent_state.rollout_refresh_records.append({
            "refresh_state": torch.zeros(4, dtype=torch.float32),
            "opponent_id": 2,
            "reason_id": 1,
            "flag_state_bucket": 2,
            "carrier_progress_bucket": 1,
            "next_z": 1,
            "return_at_refresh": 0.0,
        })
        latent_state.rollout_strategy_episode_records.append({
            "episode_id": 0,
            "global_state_0": torch.zeros(4, dtype=torch.float32),
            "z": 1,
            "z_logprob_old": 0.0,
            "episode_return": 15.0,
            "bucket_id": 5,
            "opponent_id": 2,
            "q_phi_probs": [0.25] * 4,
        })

        stats = latent_state.apply_episode_strategy_ppo(latent_lam_h=0.0)

        self.assertEqual(stats["latent_v3i3_event_pref_active_records"], 1.0)
        self.assertEqual(stats["latent_v3i3_event_pref_fallback_oef"], 1.0)
        self.assertEqual(stats["latent_v3i3_event_pref_fallback_full"], 0.0)
        self.assertAlmostEqual(
            stats["latent_v3i3_event_pref_target_entropy"], 1.38629436, places=4
        )

    def test_apply_episode_strategy_ppo_v3i7_awrd_uses_winning_z_margin(self) -> None:
        trainer = _make_trainer(n_envs=1, warmup=0, episode_credit=True, gs_dim=4)
        trainer.cfg = SimpleNamespace(opponent_pool=["OP5"], opponent_pool_weights=[1.0])
        trainer.global_step = 100
        trainer.latent_k = 4
        trainer.latent_episode_strategy_coef = 0.0
        trainer.latent_episode_strategy_n_epochs = 1
        trainer.cfg.max_grad_norm = 1.0
        trainer.latent_episode_strategy_return_norm = False
        trainer.latent_episode_strategy_value_coef = 0.5
        trainer.latent_entropy_objective = "maximize"
        trainer.latent_episode_strategy_clip_eps = 0.2
        trainer.latent_lam_h = 0.0
        trainer.latent_preference_coef = 0.0
        trainer.latent_v3i3_event_preference_enabled = False
        trainer.latent_awrd_enabled = True
        trainer.latent_awrd_coef = 0.5
        trainer.latent_awrd_temperature = 0.35
        trainer.latent_awrd_min_bucket_count = 4
        trainer.latent_awrd_min_distinct_z = 2
        trainer.latent_awrd_margin_threshold = 0.15
        trainer.latent_awrd_margin_scale = 2.0

        class MockOptimizer:
            def __init__(self):
                self.zero_grad_called = False
                self.step_called = False
                self.param_groups = [{"params": []}]
            def zero_grad(self, set_to_none=True):
                self.zero_grad_called = True
            def step(self):
                self.step_called = True

        trainer.optimizer = MockOptimizer()
        trainer.latent_router_optimizer = None

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.global_state_dim = 4
                self.strategy_encoder = torch.nn.Linear(4, 4)
                self.episode_strategy_value_head = torch.nn.Linear(4, 1)
            def strategy_logits(self, state):
                return self.strategy_encoder(state)
            def episode_strategy_value(self, state, z):
                return self.episode_strategy_value_head(state).squeeze(-1)

        trainer.model = MockModel()

        latent_state = LatentStrategyState(trainer)
        latent_state.reset()
        for z_val, win_loss in ((2, 1), (2, 1), (3, 0), (3, 1)):
            latent_state.latent_preference_buffer.append({
                "context_bucket": 5,
                "opponent": 4,
                "phase_flag_state": 5,
                "z": z_val,
                "return": float(win_loss),
                "behavior_embedding": [0.0] * 13,
                "win_loss": win_loss,
            })

        latent_state.rollout_strategy_episode_records.append({
            "episode_id": 0,
            "global_state_0": torch.zeros(4, dtype=torch.float32),
            "z": 2,
            "z_logprob_old": 0.0,
            "episode_return": 1.0,
            "bucket_id": 5,
            "opponent_id": 4,
            "q_phi_probs": [0.25] * 4,
        })

        stats = latent_state.apply_episode_strategy_ppo(latent_lam_h=0.0)

        self.assertGreater(stats["latent_awrd_loss"], 0.0)
        self.assertEqual(stats["latent_awrd_coef_scale"], 1.0)
        self.assertEqual(stats["latent_awrd_active_fraction"], 1.0)
        self.assertEqual(stats["latent_awrd_active_buckets"], 1.0)
        self.assertAlmostEqual(stats["latent_awrd_margin_mean"], 0.5, places=5)
        self.assertAlmostEqual(stats["latent_awrd_wr_spread_mean"], 0.5, places=5)
        self.assertEqual(stats["latent_awrd_best_z_mean"], 2.0)

    def test_strict_faithful_leakage_prevention(self) -> None:
        """Verify that telemetry, opponent, and phase fields are NOT read inside _policy_z_separation_loss."""
        from rl.custom_ppo.ppo_updater import _policy_z_separation_loss, StrictFaithfulDictWrapper

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n_agents = 2
                self.per_agent_action_dims = [3, 3]
            def _mask_logits(self, logits, mask):
                return logits
            def policy_logits(self, obs, z_idx=None):
                batch_size = z_idx.shape[0] if z_idx is not None else 1
                return torch.zeros((batch_size, 6))

        model = MockModel()

        obs_batch = {
            "grid": torch.zeros((4, 2, 5, 5, 5)),
            "vec": torch.zeros((4, 2, 10)),
            "agent_mask": torch.ones((4, 2)),
            "mask": torch.ones((4, 6)),
        }

        forbidden_keys = [
            "opponent_id", "phase_id", "phase", "outcome_id", "role_bucket_id",
            "spread_bucket_id", "pressure_bucket_id", "attack_defense_ratio_bucket_id",
            "role_bucket", "spread_bucket", "pressure_bucket", "attack_defense_ratio_bucket",
            "opponent", "outcome"
        ]
        for key in forbidden_keys:
            obs_batch[key] = torch.ones((4,))

        z_idx = torch.zeros((4,), dtype=torch.long)

        loss, stats = _policy_z_separation_loss(
            model,
            obs_batch,
            z_idx,
            latent_k=4,
            margin=0.08,
        )
        self.assertIsNotNone(loss)
        self.assertIn("jsd", stats)

        wrapped_obs = StrictFaithfulDictWrapper(obs_batch)
        for key in forbidden_keys:
            with self.assertRaises(AssertionError):
                _ = wrapped_obs[key]
            with self.assertRaises(AssertionError):
                _ = wrapped_obs.get(key)


if __name__ == "__main__":
    unittest.main()
