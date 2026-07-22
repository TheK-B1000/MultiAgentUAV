import os
import tempfile
import unittest
from unittest import mock

from rl.roastar_league import ROAStarLeague
from rl.train_ppo_roastar import ExploiterTriggerCallback, find_latest_snapshot


class _FakeModel:
    def __init__(self) -> None:
        self.saved_paths = []

    def save(self, path: str) -> None:
        self.saved_paths.append(path)


class _FakeCfg:
    max_blue_agents = 2
    device = "cpu"
    seed = 1
    checkpoint_dir = ""
    run_tag = "unit_test_run"


class ExploiterLifecycleTests(unittest.TestCase):
    """
    Verifies the Stage 4/3B lifecycle end-to-end at the orchestration level,
    without actually running PPO training: main snapshot -> exploiter training
    -> exploiter registration -> subsequent PFSP sampling can select it.
    train_attacker_exploiter is mocked out (real training is already covered by
    the smoke test in tests/test_exploiter_env.py and the manual CPU run).
    """

    def test_exploiter_lifecycle_registers_into_league_and_becomes_sampleable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _FakeCfg()
            cfg.checkpoint_dir = tmpdir

            # snapshot_prob=1.0/anchor_op3_prob=0/species_prob=0 forces the SNAPSHOT
            # category deterministically, so a single sample is enough to prove
            # the newly-registered exploiter is reachable by PFSP sampling.
            league = ROAStarLeague(seed=1, species_prob=0.0, snapshot_prob=1.0, anchor_op3_prob=0.0)
            fake_exploiter_path = os.path.join(tmpdir, "exploiter001.zip")
            league_state_path = os.path.join(tmpdir, "league_state.json")

            with mock.patch(
                "rl.train_ppo_roastar.train_attacker_exploiter",
                return_value=fake_exploiter_path,
            ) as mocked_train:
                callback = ExploiterTriggerCallback(
                    cfg=cfg,
                    league=league,
                    every_steps=100,
                    every_episodes=None,
                    exploiter_total_steps=1234,
                    exploiter_n_envs=8,
                    league_state_path=league_state_path,
                )
                callback.model = _FakeModel()
                callback._run_exploiter_cycle()

                mocked_train.assert_called_once()
                _, kwargs = mocked_train.call_args
                self.assertTrue(kwargs["blue_snapshot_path"].endswith(".zip"))
                self.assertEqual(kwargs["n_agents"], cfg.max_blue_agents)

            # Step 1: main snapshot was frozen before exploiter training started.
            self.assertEqual(len(callback.model.saved_paths), 1)

            # Steps 3: exploiter checkpoint registered into the PFSP pool, tagged
            # as exploiter-origin (not indistinguishable from a plain self-play snapshot).
            expected_path = os.path.abspath(fake_exploiter_path)
            self.assertIn(expected_path, league.snapshots)
            self.assertIn(expected_path, league.exploiter_snapshots)

            # Persisted state so a resumed process wouldn't forget this.
            self.assertTrue(os.path.isfile(league_state_path))

            # Step 4: subsequent PFSP sampling can select the newly-registered exploiter.
            spec = league.sample_league_pfsp(phase="OP3", enable_snapshots=True)
            self.assertEqual(spec.kind, "SNAPSHOT")
            self.assertEqual(spec.key, expected_path)

    def test_trigger_fires_only_after_step_threshold(self):
        cfg = _FakeCfg()
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.checkpoint_dir = tmpdir
            league = ROAStarLeague(seed=2)
            callback = ExploiterTriggerCallback(
                cfg=cfg,
                league=league,
                every_steps=1000,
                every_episodes=None,
                exploiter_total_steps=10,
                exploiter_n_envs=2,
                league_state_path=None,
            )
            callback.model = _FakeModel()
            callback.locals = {"dones": [False]}

            with mock.patch.object(callback, "_run_exploiter_cycle") as mocked_cycle:
                callback.num_timesteps = 500
                callback._on_step()
                mocked_cycle.assert_not_called()

                callback.num_timesteps = 1000
                callback._on_step()
                mocked_cycle.assert_called_once()


class FindLatestSnapshotTests(unittest.TestCase):
    def test_picks_highest_episode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tag = "ppo_roastar_pfsp_2v2_seed42"
            paths = [
                os.path.join(tmpdir, f"{tag}_league_snapshot_ep030200.zip"),
                os.path.join(tmpdir, f"{tag}_league_snapshot_ep030600.zip"),
                os.path.join(tmpdir, f"{tag}_league_snapshot_ep030400.zip"),
            ]
            for p in paths:
                open(p, "wb").close()
            got = find_latest_snapshot(tmpdir, tag)
            self.assertEqual(got, paths[1])

    def test_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertIsNone(find_latest_snapshot(tmpdir, "no_such_tag"))


if __name__ == "__main__":
    unittest.main()
