"""Periodic checkpoint schedule: no catch-up on resume."""

from __future__ import annotations

import unittest

from rl.custom_ppo.checkpoint_schedule import (
    CheckpointSchedule,
    advance_checkpoint_schedule,
    checkpoint_due,
    resolve_checkpoint_schedule,
)


class CheckpointScheduleTests(unittest.TestCase):
    def test_stage2_resume_uses_run_relative_next_50k(self) -> None:
        schedule = resolve_checkpoint_schedule(
            global_step=1_048_576,
            interval=50_000,
            checkpoint_run_start_step=1_048_576,
            additional_timesteps=500_000,
            load_weights_only=True,
        )
        self.assertEqual(schedule.mode, "run_progress")
        self.assertEqual(schedule.run_start_step, 1_048_576)
        self.assertEqual(schedule.next_progress, 50_000)
        self.assertIsNone(checkpoint_due(schedule, 1_048_576))
        self.assertIsNone(checkpoint_due(schedule, 1_098_575))
        self.assertEqual(checkpoint_due(schedule, 1_098_576), 50_000)

    def test_stage2_no_catch_up_burst_at_resume(self) -> None:
        schedule = resolve_checkpoint_schedule(
            global_step=1_048_576,
            interval=50_000,
            checkpoint_run_start_step=1_048_576,
            additional_timesteps=500_000,
            load_weights_only=True,
        )
        labels: list[int] = []
        gs = 1_048_576
        while gs <= 1_100_000:
            label = checkpoint_due(schedule, gs)
            if label is not None:
                labels.append(label)
                advance_checkpoint_schedule(schedule)
            gs += 25_000
        self.assertEqual(labels, [50_000])

    def test_global_resume_skips_missed_thresholds(self) -> None:
        schedule = resolve_checkpoint_schedule(
            global_step=1_048_576,
            interval=50_000,
            checkpoint_run_start_step=0,
            additional_timesteps=0,
            load_weights_only=False,
        )
        self.assertEqual(schedule.mode, "global")
        self.assertEqual(schedule.next_global_step, 1_050_000)
        self.assertIsNone(checkpoint_due(schedule, 1_048_576))
        self.assertEqual(checkpoint_due(schedule, 1_050_000), 1_050_000)
        advance_checkpoint_schedule(schedule)
        self.assertEqual(schedule.next_global_step, 1_100_000)
        self.assertIsNone(checkpoint_due(schedule, 1_075_000))

    def test_fresh_run_global_first_at_interval(self) -> None:
        schedule = resolve_checkpoint_schedule(
            global_step=0,
            interval=50_000,
            checkpoint_run_start_step=0,
            additional_timesteps=0,
            load_weights_only=False,
        )
        self.assertEqual(schedule.next_global_step, 50_000)
        self.assertEqual(checkpoint_due(schedule, 50_000), 50_000)

    def test_disabled_when_interval_zero(self) -> None:
        schedule = resolve_checkpoint_schedule(
            global_step=100,
            interval=0,
            checkpoint_run_start_step=0,
            additional_timesteps=500_000,
            load_weights_only=True,
        )
        self.assertEqual(schedule.mode, "disabled")
        self.assertIsNone(checkpoint_due(schedule, 1_000_000))


if __name__ == "__main__":
    unittest.main()
