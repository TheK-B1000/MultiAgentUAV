import os
import tempfile
import unittest

from game_field_gpu import BatchedCTFCore, GPUFieldConfig


class SnapshotCacheEvictionTests(unittest.TestCase):
    """
    Regression test for a memory leak: _snapshot_policy_cache only ever grew
    (every unique snapshot path ever loaded stayed cached forever, even after
    LeagueCallback._enforce_league_snapshot_limit deleted the file on disk),
    which slowly accumulated full loaded SB3 models in host RAM over long
    league-mode runs and contributed to an OOM crash in production ablation
    training. _prune_stale_snapshot_cache must remove entries whose backing
    file no longer exists.
    """

    def _make_core(self) -> BatchedCTFCore:
        cfg = GPUFieldConfig(n_envs=1, max_blue_agents=2, max_red_agents=2, device="cpu", seed=1)
        return BatchedCTFCore(cfg)

    def test_prune_removes_entries_for_deleted_files(self):
        core = self._make_core()
        with tempfile.TemporaryDirectory() as tmpdir:
            still_here = os.path.join(tmpdir, "still_here.zip")
            deleted = os.path.join(tmpdir, "deleted.zip")
            open(still_here, "wb").close()
            open(deleted, "wb").close()

            core._snapshot_policy_cache[still_here] = (1.0, object())
            core._snapshot_policy_cache[deleted] = (1.0, object())

            os.remove(deleted)
            core._prune_stale_snapshot_cache()

            self.assertIn(still_here, core._snapshot_policy_cache)
            self.assertNotIn(deleted, core._snapshot_policy_cache)

    def test_prune_is_a_noop_when_all_files_still_exist(self):
        core = self._make_core()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "a.zip")
            open(path, "wb").close()
            core._snapshot_policy_cache[path] = (1.0, object())

            core._prune_stale_snapshot_cache()

            self.assertIn(path, core._snapshot_policy_cache)

    def test_load_snapshot_policy_prunes_stale_entries_before_lookup(self):
        core = self._make_core()
        with tempfile.TemporaryDirectory() as tmpdir:
            deleted = os.path.join(tmpdir, "deleted.zip")
            open(deleted, "wb").close()
            core._snapshot_policy_cache[deleted] = (1.0, object())
            os.remove(deleted)

            # Trigger path must resolve (exist) to reach the prune step; its
            # contents are irrelevant here (an invalid/empty zip just makes the
            # subsequent SB3 load fail and return None, which is fine).
            trigger = os.path.join(tmpdir, "trigger.zip")
            open(trigger, "wb").close()

            core._load_snapshot_policy(trigger)

            self.assertNotIn(deleted, core._snapshot_policy_cache)


if __name__ == "__main__":
    unittest.main()
