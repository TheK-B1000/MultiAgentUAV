import os
import unittest

from validate_checkpoints import read_metadata, validate_one

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRACKED_CHECKPOINT = os.path.join(_PROJECT_ROOT, "checkpoints_sb3", "2v2", "final_ppo_league_2v2.zip")


class ReadMetadataTests(unittest.TestCase):
    @unittest.skipUnless(os.path.isfile(_TRACKED_CHECKPOINT), "requires tracked checkpoint")
    def test_reads_real_checkpoint_metadata(self):
        num_timesteps, total_target, errors = read_metadata(_TRACKED_CHECKPOINT)
        self.assertEqual(errors, [])
        self.assertIsInstance(num_timesteps, int)
        self.assertGreater(num_timesteps, 0)
        self.assertEqual(total_target, 1_000_000)

    def test_missing_file_reports_error_not_exception(self):
        num_timesteps, total_target, errors = read_metadata("/does/not/exist.zip")
        self.assertIsNone(num_timesteps)
        self.assertIsNone(total_target)
        self.assertTrue(errors)

    def test_non_zip_file_reports_error_not_exception(self):
        # This test file itself is not a zip -- exercises the BadZipFile path.
        num_timesteps, total_target, errors = read_metadata(os.path.abspath(__file__))
        self.assertIsNone(num_timesteps)
        self.assertTrue(errors)


class ValidateOneTests(unittest.TestCase):
    def test_missing_checkpoint_fails_gracefully(self):
        report = validate_one("/does/not/exist.zip", n_agents=2, device="cpu", metadata_only=True)
        self.assertFalse(report.ok)
        self.assertIn("file not found", report.errors)

    @unittest.skipUnless(os.path.isfile(_TRACKED_CHECKPOINT), "requires tracked checkpoint")
    def test_metadata_only_on_real_checkpoint_is_ok_and_fast(self):
        report = validate_one(_TRACKED_CHECKPOINT, n_agents=2, device="cpu", metadata_only=True)
        self.assertTrue(report.ok)
        self.assertFalse(report.deep_checked)
        self.assertIsNotNone(report.progress_pct)


if __name__ == "__main__":
    unittest.main()
