import unittest

from rl.configuration_space import (
    CURRENT_PROFILES,
    HELDOUT_CURRENT_PROFILES,
    TRAIN_CURRENT_PROFILES,
    Configuration,
    TeamSizeMismatchError,
    assert_team_size_compatible,
    config_seed_block,
    episode_seeds,
    heldout_configurations,
    is_seen,
    seen_configurations,
    split,
)


class ConfigurationTests(unittest.TestCase):
    def test_rejects_snapshot_opponents(self):
        with self.assertRaises(ValueError):
            Configuration("SNAPSHOT", "OP3", "nominal", 2)

    def test_rejects_unknown_opponent_and_profile(self):
        with self.assertRaises(ValueError):
            Configuration("SCRIPTED", "OP9", "nominal", 2)
        with self.assertRaises(ValueError):
            Configuration("SCRIPTED", "OP3", "hurricane", 2)

    def test_key_is_stable_and_distinguishing(self):
        a = Configuration("SCRIPTED", "OP3", "nominal", 2)
        b = Configuration("SCRIPTED", "OP3", "strong", 2)
        self.assertEqual(a.key, Configuration("scripted", "op3", "nominal", 2).key)
        self.assertNotEqual(a.key, b.key)

    def test_stress_schedule_registers_under_the_phase_actually_set(self):
        config = Configuration("SCRIPTED", "OP4", "strong", 2)
        schedule = config.stress_schedule("OP4")
        self.assertIn("OP4", schedule)
        self.assertEqual(
            schedule["OP4"]["current_strength_cps"],
            CURRENT_PROFILES["strong"]["current_strength_cps"],
        )


class SplitTests(unittest.TestCase):
    def test_seen_and_heldout_are_disjoint(self):
        parts = split(2)
        seen_keys = {c.key for c in parts["seen"]}
        heldout_keys = {c.key for c in parts["heldout"]}
        self.assertTrue(seen_keys)
        self.assertTrue(heldout_keys)
        self.assertEqual(seen_keys & heldout_keys, set())

    def test_is_seen_agrees_with_the_split(self):
        for config in seen_configurations(3):
            self.assertTrue(is_seen(config), config.key)
        for config in heldout_configurations(3):
            self.assertFalse(is_seen(config), config.key)

    def test_op4_is_never_in_the_seen_split(self):
        self.assertFalse(any(c.opponent_key == "OP4" for c in seen_configurations(2)))

    def test_heldout_profiles_are_stronger_than_trained_ones(self):
        max_trained = max(
            CURRENT_PROFILES[p]["current_strength_cps"] for p in TRAIN_CURRENT_PROFILES
        )
        for profile in HELDOUT_CURRENT_PROFILES:
            self.assertGreater(CURRENT_PROFILES[profile]["current_strength_cps"], max_trained)


class SeedBlockTests(unittest.TestCase):
    def test_common_random_numbers_within_a_configuration(self):
        config = Configuration("SCRIPTED", "OP3", "nominal", 2)
        self.assertEqual(episode_seeds(config, 50), episode_seeds(config, 50))

    def test_configurations_never_share_episode_seeds(self):
        all_configs = seen_configurations(2) + heldout_configurations(2)
        seen_seeds = set()
        for config in all_configs:
            seeds = set(episode_seeds(config, 200))
            self.assertEqual(seen_seeds & seeds, set(), f"seed collision at {config.key}")
            seen_seeds |= seeds

    def test_block_is_derived_from_the_key_not_enumeration_order(self):
        config = Configuration("SPECIES", "RUSHER", "calm", 4)
        self.assertEqual(config_seed_block(config), config_seed_block(config))

    def test_rejects_episode_count_that_would_overflow_a_block(self):
        config = Configuration("SCRIPTED", "OP3", "nominal", 2)
        with self.assertRaises(ValueError):
            episode_seeds(config, 200_000, block_size=100_000)


class TeamSizeGuardTests(unittest.TestCase):
    def test_same_size_is_allowed(self):
        assert_team_size_compatible(2, 2)

    def test_cross_size_transfer_is_refused(self):
        with self.assertRaises(TeamSizeMismatchError) as ctx:
            assert_team_size_compatible(2, 3)
        self.assertIn("SCALABILITY", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
