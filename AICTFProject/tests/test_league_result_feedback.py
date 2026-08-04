"""Regression guard for the bug where no match results ever reached the league.

Until 2026-08 ``ROAStarLeagueCallback`` overrode opponent *selection* only, and
nothing called ``league.record_result()``. ``win_rate()`` returned None for every
opponent, ``_pfsp_weight()`` returned its unplayed value of 1.0, and ``--mode
pfsp`` silently degenerated to a uniform draw. Every 2v2/3v3/4v4 run finished
before the fix has an empty ``win_rate_stats``.

These tests pin the feedback path itself, without needing a GPU or a real
training loop.
"""
import unittest

from rl.egt_league import DoubleOracleLeague, FictitiousPlayLeague
from rl.roastar_league import ROAStarLeague
from rl.train_ppo_roastar import (
    LEAGUE_MODES,
    DoubleOracleLeagueCallback,
    FictitiousPlayLeagueCallback,
    ROAStarLeagueCallback,
    _EGTLeagueCallbackBase,
)
from rl.verify_league_runs import LeagueRunAudit


class _StubCallback(_EGTLeagueCallbackBase):
    """Exercises _update_opponent_stats without constructing a real SB3 callback."""

    def __init__(self, league):
        self.league = league
        self._enable_opponent_tracking = False

    def _select_league_opponent(self):
        raise AssertionError("not exercised by these tests")


class ResultFeedbackTests(unittest.TestCase):
    def test_results_reach_the_league(self):
        league = ROAStarLeague(seed=1)
        cb = _StubCallback(league)

        cb._update_opponent_stats("SCRIPTED:OP3", "WIN")
        cb._update_opponent_stats("SCRIPTED:OP3", "LOSS")
        cb._update_opponent_stats("SCRIPTED:OP3", "DRAW")

        self.assertEqual(league.win_rate("SCRIPTED:OP3"), (1.0 + 0.0 + 0.5) / 3)

    def test_recording_happens_even_when_callback_tracking_is_disabled(self):
        # _enable_opponent_tracking=False makes LeagueCallback's own bookkeeping
        # early-return; the league's stats must not depend on that flag.
        league = ROAStarLeague(seed=1)
        cb = _StubCallback(league)
        cb._enable_opponent_tracking = False

        cb._update_opponent_stats("SPECIES:RUSHER", "WIN")

        self.assertEqual(league.win_rate("SPECIES:RUSHER"), 1.0)

    def test_pfsp_weight_actually_moves_once_results_arrive(self):
        """The whole point: without feedback every weight is the 1.0 unplayed value."""
        league = ROAStarLeague(seed=1, pfsp_p=2.0)
        cb = _StubCallback(league)

        self.assertEqual(league._pfsp_weight("SCRIPTED:OP3"), 1.0)  # unplayed
        for _ in range(10):
            cb._update_opponent_stats("SCRIPTED:OP3", "WIN")
        self.assertLess(league._pfsp_weight("SCRIPTED:OP3"), 1.0)

    def test_unknown_result_strings_are_ignored_not_miscounted(self):
        league = ROAStarLeague(seed=1)
        cb = _StubCallback(league)
        cb._update_opponent_stats("SCRIPTED:OP1", "TIMEOUT")
        self.assertIsNone(league.win_rate("SCRIPTED:OP1"))

    def test_double_oracle_payoff_matrix_is_fed_too(self):
        league = DoubleOracleLeague(seed=1, min_games_for_payoff=1)
        cb = _StubCallback(league)
        cb._update_opponent_stats("SCRIPTED:OP2", "WIN")
        self.assertEqual(league.payoff_stats[(league.learner_key, "SCRIPTED:OP2")], (1.0, 1))


class ModeWiringTests(unittest.TestCase):
    def test_every_mode_pairs_a_league_with_a_feedback_callback(self):
        expected = {
            "pfsp": (ROAStarLeague, ROAStarLeagueCallback),
            "pfsp_exploiter": (ROAStarLeague, ROAStarLeagueCallback),
            "fp": (FictitiousPlayLeague, FictitiousPlayLeagueCallback),
            "do": (DoubleOracleLeague, DoubleOracleLeagueCallback),
        }
        self.assertEqual(LEAGUE_MODES, expected)

    def test_all_callbacks_inherit_the_feedback_path(self):
        for _mode, (_league_cls, cb_cls) in LEAGUE_MODES.items():
            self.assertTrue(issubclass(cb_cls, _EGTLeagueCallbackBase))


class AuditTests(unittest.TestCase):
    def _audit(self, **kwargs):
        base = dict(
            path="x.json",
            mode="pfsp",
            setting="2v2",
            seed=42,
            n_snapshots=3,
            n_opponents_with_results=0,
            n_exploiter_snapshots=0,
            n_payoff_entries=0,
        )
        base.update(kwargs)
        return LeagueRunAudit(**base)

    def test_empty_stats_fails_a_pfsp_run(self):
        audit = self._audit(mode="pfsp", n_opponents_with_results=0)
        self.assertFalse(audit.ok)
        self.assertTrue(any("UNIFORM" in r for r in audit.failures))

    def test_populated_stats_passes_a_pfsp_run(self):
        self.assertTrue(self._audit(mode="pfsp", n_opponents_with_results=6).ok)

    def test_fictitious_play_does_not_need_result_feedback(self):
        # FP samples uniformly by definition, so empty stats is not a defect.
        self.assertTrue(self._audit(mode="fp", n_opponents_with_results=0).ok)

    def test_double_oracle_needs_a_populated_payoff_matrix(self):
        audit = self._audit(mode="do", n_opponents_with_results=6, n_payoff_entries=0)
        self.assertFalse(audit.ok)
        self.assertTrue(any("payoff" in r for r in audit.failures))

    def test_exploiter_mode_requires_exploiters_in_the_pool(self):
        audit = self._audit(
            mode="pfsp_exploiter", n_opponents_with_results=6, n_exploiter_snapshots=0
        )
        self.assertFalse(audit.ok)
        self.assertTrue(any("exploiter" in r for r in audit.failures))

    def test_empty_snapshot_pool_fails_any_mode(self):
        audit = self._audit(mode="fp", n_snapshots=0)
        self.assertFalse(audit.ok)


if __name__ == "__main__":
    unittest.main()
