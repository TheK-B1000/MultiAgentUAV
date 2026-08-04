import unittest
from collections import Counter

import numpy as np

from rl.egt_league import DoubleOracleLeague, FictitiousPlayLeague, solve_zero_sum_nash
from rl.roastar_league import ROAStarLeague


def _league(cls, **kwargs):
    """Snapshot-only league so the category draw always lands on SNAPSHOT."""
    return cls(seed=1, species_prob=0.0, snapshot_prob=1.0, anchor_op3_prob=0.0, **kwargs)


class NashSolverTests(unittest.TestCase):
    def test_rock_paper_scissors_is_uniform(self):
        rps = np.array([[0.5, 0.0, 1.0], [1.0, 0.5, 0.0], [0.0, 1.0, 0.5]], dtype=float)
        x = solve_zero_sum_nash(rps)
        np.testing.assert_allclose(x, np.full(3, 1.0 / 3.0), atol=1e-6)

    def test_dominant_strategy_takes_all_mass(self):
        # Row 0 beats every column; row 1 loses to every column.
        payoff = np.array([[0.5, 1.0], [0.0, 0.5]], dtype=float)
        x = solve_zero_sum_nash(payoff)
        self.assertGreater(x[0], 0.99)

    def test_returns_a_probability_vector(self):
        rng = np.random.default_rng(0)
        payoff = rng.random((5, 5))
        x = solve_zero_sum_nash(payoff)
        self.assertAlmostEqual(float(x.sum()), 1.0, places=6)
        self.assertTrue(bool((x >= -1e-9).all()))

    def test_single_row_is_degenerate(self):
        np.testing.assert_allclose(solve_zero_sum_nash(np.array([[0.5]])), [1.0])

    def test_rejects_empty_matrix(self):
        with self.assertRaises(ValueError):
            solve_zero_sum_nash(np.zeros((0, 0)))


class FictitiousPlayLeagueTests(unittest.TestCase):
    def test_sampling_is_uniform_regardless_of_win_rate(self):
        league = _league(FictitiousPlayLeague, min_episodes_per_opponent=1)
        league.snapshots = ["easy", "hard"]
        # Lopsided results must NOT bias FP -- that is what makes it FP and not PFSP.
        for _ in range(50):
            league.record_result("SNAPSHOT:easy", 1.0)
            league.record_result("SNAPSHOT:hard", 0.0)

        counts = Counter(
            league.sample_league_fp(phase="OP3", enable_snapshots=True).key for _ in range(4000)
        )
        self.assertGreater(counts["easy"], 0)
        self.assertGreater(counts["hard"], 0)
        ratio = counts["easy"] / counts["hard"]
        self.assertGreater(ratio, 0.8)
        self.assertLess(ratio, 1.25)

    def test_differs_from_pfsp_on_the_same_history(self):
        pfsp = _league(ROAStarLeague, min_episodes_per_opponent=1, pfsp_p=2.0)
        pfsp.snapshots = ["easy", "hard"]
        for _ in range(50):
            pfsp.record_result("SNAPSHOT:easy", 1.0)
            pfsp.record_result("SNAPSHOT:hard", 0.0)
        pfsp_counts = Counter(
            pfsp.sample_league_pfsp(phase="OP3", enable_snapshots=True).key for _ in range(4000)
        )
        # PFSP concentrates on the opponent it loses to; FP (tested above) does not.
        self.assertGreater(pfsp_counts["hard"], 2 * pfsp_counts["easy"])

    def test_keeps_the_category_split_and_stickiness(self):
        league = FictitiousPlayLeague(seed=3, min_episodes_per_opponent=3)
        league.snapshots = ["a", "b"]
        first = league.sample_league_fp(phase="OP3", enable_snapshots=True)
        second = league.sample_league_fp(phase="OP3", enable_snapshots=True)
        self.assertEqual((first.kind, first.key), (second.kind, second.key))


class DoubleOracleLeagueTests(unittest.TestCase):
    def test_record_result_fills_the_learner_payoff_row(self):
        league = _league(DoubleOracleLeague, min_games_for_payoff=1)
        league.record_result("SCRIPTED:OP3", 1.0)
        league.record_result("SCRIPTED:OP3", 0.0)
        total, games = league.payoff_stats[(league.learner_key, "SCRIPTED:OP3")]
        self.assertEqual((total, games), (1.0, 2))
        # PFSP stats stay in sync so the two rules remain comparable.
        self.assertEqual(league.win_rate("SCRIPTED:OP3"), 0.5)

    def test_add_snapshot_freezes_the_learner_row(self):
        league = _league(DoubleOracleLeague, min_games_for_payoff=1)
        for _ in range(4):
            league.record_result("SCRIPTED:OP1", 1.0)
        league.add_snapshot("checkpoints_sb3/2v2/snap_ep100.zip")
        self.assertIn(("SNAPSHOT:snap_ep100", "SCRIPTED:OP1"), league.payoff_stats)

    def test_payoff_matrix_uses_the_zero_sum_reflection(self):
        league = _league(DoubleOracleLeague, min_games_for_payoff=1)
        league._add_payoff("A", "B", 1.0)
        league._add_payoff("A", "B", 1.0)
        mat = league.payoff_matrix(["A", "B"])
        self.assertAlmostEqual(mat[0, 1], 1.0)
        self.assertAlmostEqual(mat[1, 0], 0.0)  # reflected, never observed directly
        self.assertAlmostEqual(mat[0, 0], 0.5)

    def test_unobserved_matchups_default_to_even(self):
        league = _league(DoubleOracleLeague, min_games_for_payoff=5)
        league._add_payoff("A", "B", 1.0)  # only 1 game, below the threshold
        mat = league.payoff_matrix(["A", "B"])
        self.assertAlmostEqual(mat[0, 1], 0.5)

    def test_meta_nash_is_a_distribution_over_the_pool(self):
        league = _league(DoubleOracleLeague, min_games_for_payoff=1)
        league.add_snapshot("checkpoints_sb3/2v2/snap_ep100.zip")
        nash = league.meta_nash(enable_snapshots=True)
        self.assertIn("SNAPSHOT:snap_ep100", nash)
        self.assertAlmostEqual(sum(nash.values()), 1.0, places=6)

    def test_sampling_favours_the_meta_nash_support(self):
        league = _league(DoubleOracleLeague, min_games_for_payoff=1, min_episodes_per_opponent=1)
        league.add_snapshot("checkpoints_sb3/2v2/strong.zip")
        league.add_snapshot("checkpoints_sb3/2v2/weak.zip")
        # "strong" beats everything in the pool; "weak" loses to everything.
        for key in league.pool_keys():
            if key == "SNAPSHOT:strong":
                continue
            for _ in range(5):
                league._add_payoff("SNAPSHOT:strong", key, 1.0)
                league._add_payoff("SNAPSHOT:weak", key, 0.0)

        counts = Counter(
            league.sample_league_do(phase="OP3", enable_snapshots=True).key for _ in range(2000)
        )
        strong = counts["checkpoints_sb3/2v2/strong.zip"]
        weak = counts["checkpoints_sb3/2v2/weak.zip"]
        self.assertGreater(strong, weak)

    def test_state_round_trips_through_serialization(self):
        league = _league(DoubleOracleLeague, min_games_for_payoff=1)
        league.record_result("SCRIPTED:OP2", 0.5)
        restored = _league(DoubleOracleLeague, min_games_for_payoff=1)
        restored.load_state_dict(league.to_dict())
        self.assertEqual(
            restored.payoff_stats[(restored.learner_key, "SCRIPTED:OP2")], (0.5, 1)
        )
        self.assertEqual(restored.win_rate("SCRIPTED:OP2"), 0.5)


if __name__ == "__main__":
    unittest.main()
