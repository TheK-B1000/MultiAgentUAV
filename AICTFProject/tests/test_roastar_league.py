import unittest

from rl.roastar_league import ROAStarLeague


class ROAStarLeagueTests(unittest.TestCase):
    def _make_league(self, **kwargs) -> ROAStarLeague:
        return ROAStarLeague(seed=1, species_prob=0.0, snapshot_prob=1.0, anchor_op3_prob=0.0, **kwargs)

    def test_win_rate_stats_update_correctly(self):
        league = self._make_league()
        league.record_result("SPECIES:RUSHER", 1.0)  # win
        league.record_result("SPECIES:RUSHER", 0.0)  # loss
        league.record_result("SPECIES:RUSHER", 0.5)  # draw

        self.assertEqual(league.win_rate("SPECIES:RUSHER"), (1.0 + 0.0 + 0.5) / 3)
        self.assertIsNone(league.win_rate("SPECIES:UNPLAYED"))

    def test_unplayed_opponent_gets_full_pfsp_weight(self):
        league = self._make_league()
        league.record_result("SPECIES:RUSHER", 1.0)
        league.record_result("SPECIES:RUSHER", 1.0)
        league.record_result("SPECIES:RUSHER", 1.0)  # 100% win rate -> should be deprioritized

        unplayed_weight = league._pfsp_weight("SPECIES:CAMPER")
        high_winrate_weight = league._pfsp_weight("SPECIES:RUSHER")

        self.assertEqual(unplayed_weight, 1.0)
        self.assertLess(high_winrate_weight, unplayed_weight)
        self.assertGreaterEqual(high_winrate_weight, league.pfsp_floor)

    def test_pfsp_sampling_favors_low_win_rate_opponent(self):
        league = self._make_league(pfsp_p=2.0)
        # Learner beats "easy" almost always, and loses to "hard" almost always.
        for _ in range(20):
            league.record_result("SNAPSHOT:easy", 1.0)
            league.record_result("SNAPSHOT:hard", 0.0)
        league.snapshots = ["easy", "hard"]
        league.ratings.setdefault("SNAPSHOT:easy", 1200.0)
        league.ratings.setdefault("SNAPSHOT:hard", 1200.0)

        counts = {"easy": 0, "hard": 0}
        for _ in range(2000):
            picked = league._weighted_pick_pfsp(league.snapshots, key_to_stats_key=lambda p: f"SNAPSHOT:{p}")
            counts[picked] += 1

        self.assertGreater(counts["hard"], counts["easy"])

    def test_register_exploiter_snapshot_tracks_metadata_and_joins_pool(self):
        league = self._make_league()
        league.register_exploiter_snapshot("checkpoints_sb3/2v2/exploiter_ep000100.zip")

        self.assertIn("checkpoints_sb3/2v2/exploiter_ep000100.zip", league.snapshots)
        self.assertIn("checkpoints_sb3/2v2/exploiter_ep000100.zip", league.exploiter_snapshots)

    def test_sample_league_pfsp_respects_stickiness(self):
        league = self._make_league(min_episodes_per_opponent=3)
        league.snapshots = ["a", "b"]
        first = league.sample_league_pfsp(phase="OP3", enable_snapshots=True)
        second = league.sample_league_pfsp(phase="OP3", enable_snapshots=True)

        self.assertEqual(first.kind, second.kind)
        self.assertEqual(first.key, second.key)


if __name__ == "__main__":
    unittest.main()
