"""Audited strategic-niche opponent tag discipline for Summer / LRO."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from gpu_env._core._bt_profiles import (
    LRO_AUDITED_OPPONENT_POOL,
    OPPONENT_ALIASES,
    canonicalize_opponent_key,
    normalize_bt_level,
    profile_for_level,
    profile_for_opponent_key,
    role_gate_fingerprint,
)
from gpu_env._core._scripted_red import bt_dispatch_level_for_opponent_key
from rl.config.ppo_config import PPOConfig, TrainMode
from rl.training.config_validation import normalize_and_validate_training_config


EXPECTED_ROLE_GATES = {
    # (escort, counter, counter_always, mines, 2v1, intercept)
    6: (False, False, False, False, False, False),  # dual-assault TURTLE host
    7: (False, False, False, True, False, True),
    8: (True, True, False, False, True, True),
    9: (False, True, False, False, False, True),
    10: (False, False, False, False, False, True),  # pure interceptor
    11: (True, True, True, False, True, True),
    12: (True, True, True, False, False, True),
}


class OpponentAliasDisciplineTests(unittest.TestCase):
    def test_short_aliases_resolve_to_documented_long_names(self) -> None:
        self.assertEqual(OPPONENT_ALIASES["OP6"], "OP6_IMMEDIATE_DUAL_RUSH")
        self.assertEqual(OPPONENT_ALIASES["OP7"], "OP7_DEEP_FORTRESS")
        self.assertEqual(OPPONENT_ALIASES["OP8"], "OP8_PROTECTED_CARRIER_ESCORT")
        self.assertEqual(OPPONENT_ALIASES["OP9"], "OP9_SPLIT_LANE_FEINT")
        self.assertEqual(OPPONENT_ALIASES["OP10"], "OP10_AGGRESSIVE_INTERCEPTOR")
        self.assertEqual(OPPONENT_ALIASES["OP11"], "OP11_ADAPTIVE_EXPLOITER")
        self.assertEqual(OPPONENT_ALIASES["OP12"], "OP12_LATE_CONVERTER")

    def test_op8_legacy_tags_are_synonyms_for_protected_carrier_escort(self) -> None:
        self.assertEqual(canonicalize_opponent_key("OP8"), "OP8_PROTECTED_CARRIER_ESCORT")
        self.assertEqual(
            canonicalize_opponent_key("OP8_INTERCEPTOR"), "OP8_PROTECTED_CARRIER_ESCORT"
        )
        self.assertEqual(canonicalize_opponent_key("OP8_ESCORT"), "OP8_PROTECTED_CARRIER_ESCORT")
        self.assertEqual(normalize_bt_level("OP8_ESCORT"), 8)

    def test_op9_fortress_synonym_matches_audited_feint(self) -> None:
        self.assertEqual(canonicalize_opponent_key("OP9"), "OP9_SPLIT_LANE_FEINT")
        self.assertEqual(canonicalize_opponent_key("OP9_FORTRESS"), "OP9_SPLIT_LANE_FEINT")
        self.assertEqual(canonicalize_opponent_key("OP9_FEINT"), "OP9_SPLIT_LANE_FEINT")

    def test_audited_pool_has_seven_distinct_role_gates(self) -> None:
        self.assertEqual(len(LRO_AUDITED_OPPONENT_POOL), 7)
        fps = []
        for tag in LRO_AUDITED_OPPONENT_POOL:
            lvl = normalize_bt_level(tag)
            self.assertIsNotNone(lvl, tag)
            fps.append(role_gate_fingerprint(int(lvl)))
        self.assertEqual(len(set(fps)), 7, fps)

    def test_role_gate_fingerprints_pin_each_niche(self) -> None:
        for lvl, expected in EXPECTED_ROLE_GATES.items():
            with self.subTest(level=lvl):
                self.assertEqual(role_gate_fingerprint(lvl), expected)

    def test_only_one_fortress_and_one_pure_interceptor(self) -> None:
        fortress_like = []
        pure_intercept = []
        for lvl in range(6, 13):
            p = profile_for_level(lvl)
            if p.enable_mines and not p.enable_counter:
                fortress_like.append(lvl)
            if (
                p.enable_intercept
                and not p.enable_escort
                and not p.enable_counter
                and not p.enable_mines
                and not p.enable_2v1
            ):
                pure_intercept.append(lvl)
        self.assertEqual(fortress_like, [7])
        self.assertEqual(pure_intercept, [10])

    def test_short_and_long_share_profile(self) -> None:
        for short, long_name in OPPONENT_ALIASES.items():
            if short == "OP5":
                continue
            self.assertEqual(
                profile_for_opponent_key(short).name,
                profile_for_opponent_key(
                    canonicalize_opponent_key(long_name)
                ).name,
                msg=short,
            )

    def test_audited_long_names_route_to_bt_dispatch_levels(self) -> None:
        expected = {
            "OP6_IMMEDIATE_DUAL_RUSH": 6,
            "OP7_DEEP_FORTRESS": 7,
            "OP8_PROTECTED_CARRIER_ESCORT": 8,
            "OP9_SPLIT_LANE_FEINT": 9,
            "OP10_AGGRESSIVE_INTERCEPTOR": 10,
            "OP11_ADAPTIVE_EXPLOITER": 11,
            "OP12_LATE_CONVERTER": 12,
        }
        for name, level in expected.items():
            with self.subTest(name=name):
                self.assertEqual(bt_dispatch_level_for_opponent_key(name), level)

    def test_legacy_non_pool_opponents_do_not_enter_bt_dispatch(self) -> None:
        self.assertIsNone(bt_dispatch_level_for_opponent_key("OP5"))
        self.assertIsNone(bt_dispatch_level_for_opponent_key("OP5_RUSHER"))
        self.assertIsNone(bt_dispatch_level_for_opponent_key("UNKNOWN"))

    def test_training_config_accepts_audited_long_names(self) -> None:
        cfg = PPOConfig()
        cfg.mode = TrainMode.OPPONENT_POOL.value
        cfg.opponent_randomize = True
        cfg.opponent_pool = tuple(LRO_AUDITED_OPPONENT_POOL)

        out = normalize_and_validate_training_config(cfg)

        self.assertEqual(tuple(out.opponent_pool), tuple(LRO_AUDITED_OPPONENT_POOL))

    def test_niche_identities_match_table(self) -> None:
        self.assertEqual(profile_for_level(6).name, "OP6_IMMEDIATE_DUAL_RUSH")
        self.assertFalse(profile_for_level(6).enable_escort)
        self.assertFalse(profile_for_level(6).enable_intercept)
        self.assertFalse(profile_for_level(6).enable_counter)
        self.assertFalse(profile_for_level(6).counter_always)
        self.assertEqual(profile_for_level(6).min_alive_for_defender, 3)
        self.assertGreaterEqual(profile_for_level(6).lock_attacker, 20)
        self.assertLessEqual(profile_for_level(6).threat_radius, 0.5)
        self.assertGreaterEqual(profile_for_level(6).lane_amplitude_frac, 0.35)
        self.assertEqual(profile_for_level(7).name, "OP7_DEEP_FORTRESS")
        self.assertTrue(profile_for_level(7).enable_mines)
        self.assertFalse(profile_for_level(7).enable_counter)
        self.assertEqual(profile_for_level(8).name, "OP8_PROTECTED_CARRIER_ESCORT")
        self.assertTrue(profile_for_level(8).enable_escort)
        self.assertTrue(profile_for_level(8).enable_2v1)
        self.assertFalse(profile_for_level(8).counter_always)
        self.assertEqual(profile_for_level(9).name, "OP9_SPLIT_LANE_FEINT")
        self.assertGreaterEqual(profile_for_level(9).lane_amplitude_frac, 0.50)
        self.assertEqual(profile_for_level(10).name, "OP10_AGGRESSIVE_INTERCEPTOR")
        self.assertFalse(profile_for_level(10).enable_escort)
        self.assertGreaterEqual(profile_for_level(10).intercept_block_base, 0.85)
        self.assertEqual(profile_for_level(11).name, "OP11_ADAPTIVE_EXPLOITER")
        self.assertTrue(profile_for_level(11).adaptive_enabled)
        self.assertTrue(profile_for_level(11).counter_always)
        self.assertEqual(profile_for_level(12).name, "OP12_LATE_CONVERTER")
        self.assertTrue(profile_for_level(12).counter_always)
        self.assertTrue(profile_for_level(12).enable_escort)
        self.assertFalse(profile_for_level(12).enable_2v1)
        self.assertGreaterEqual(profile_for_level(12).lock_counter, 28)


if __name__ == "__main__":
    unittest.main()
