"""Unit tests for rl.behavior_telemetry bucket helpers."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from rl.behavior_telemetry import (
    N_TELEMETRY,
    ROLE2_BOTH_ATTACK,
    ROLE2_BOTH_DEFEND,
    ROLE2_ESCORT_CARRIER,
    ROLE2_INTERCEPT_ENEMY_CARRIER,
    ROLE2_SPLIT,
    ROLE4_ALL_PUSH,
    ROLE4_ESCORT_PAIR,
    ROLE4_INTERCEPT_PAIR,
    ROLE4_ONE_THREE,
    ROLE4_THREE_ONE,
    ROLE4_TWO_TWO,
    ROLE4_TURTLE,
    attack_defense_ratio_bucket_id,
    pressure_bucket_id,
    role_bucket_detailed_id,
    spread_bucket_id,
)


class BehaviorTelemetryTests(unittest.TestCase):
    def test_spread_buckets(self) -> None:
        x = np.asarray([0.02, 0.08, 0.2], dtype=np.float64)
        b = spread_bucket_id(x)
        self.assertListEqual(list(b), [0, 1, 2])

    def test_role_buckets_2v2_torch(self) -> None:
        z = torch.zeros(1)
        o = torch.ones(1)
        t = torch.tensor([2.0])
        rid = role_bucket_detailed_id(
            2,
            n_attack=t,
            n_defend=z,
            n_intercept=z,
            escort_cnt=o,
            nb_to_ec=o,
            has_blue_carrier=torch.ones(1, dtype=torch.bool),
            has_red_carrier=z.bool(),
            n_alive=t,
        )
        self.assertEqual(int(rid[0].item()), ROLE2_ESCORT_CARRIER)

        rid = role_bucket_detailed_id(
            2,
            n_attack=t,
            n_defend=z,
            n_intercept=o,
            escort_cnt=z,
            nb_to_ec=torch.tensor([0.1]),
            has_blue_carrier=z.bool(),
            has_red_carrier=torch.ones(1, dtype=torch.bool),
            n_alive=t,
        )
        self.assertEqual(int(rid[0].item()), ROLE2_INTERCEPT_ENEMY_CARRIER)

        rid = role_bucket_detailed_id(
            2,
            n_attack=t,
            n_defend=z,
            n_intercept=z,
            escort_cnt=z,
            nb_to_ec=o,
            has_blue_carrier=z.bool(),
            has_red_carrier=z.bool(),
            n_alive=t,
        )
        self.assertEqual(int(rid[0].item()), ROLE2_BOTH_ATTACK)

        rid = role_bucket_detailed_id(
            2,
            n_attack=z,
            n_defend=t,
            n_intercept=z,
            escort_cnt=z,
            nb_to_ec=o,
            has_blue_carrier=z.bool(),
            has_red_carrier=z.bool(),
            n_alive=t,
        )
        self.assertEqual(int(rid[0].item()), ROLE2_BOTH_DEFEND)

        rid = role_bucket_detailed_id(
            2,
            n_attack=o,
            n_defend=o,
            n_intercept=z,
            escort_cnt=z,
            nb_to_ec=o,
            has_blue_carrier=z.bool(),
            has_red_carrier=z.bool(),
            n_alive=t,
        )
        self.assertEqual(int(rid[0].item()), ROLE2_SPLIT)

    def test_role_buckets_4v4_torch(self) -> None:
        z = torch.zeros(1)
        o = torch.ones(1)
        tw = torch.tensor([2.0])
        fo = torch.tensor([4.0])
        rid = role_bucket_detailed_id(
            4,
            n_attack=fo,
            n_defend=z,
            n_intercept=z,
            escort_cnt=torch.tensor([2.0]),
            nb_to_ec=o,
            has_blue_carrier=torch.ones(1, dtype=torch.bool),
            has_red_carrier=z.bool(),
            n_alive=fo,
        )
        self.assertEqual(int(rid[0].item()), ROLE4_ESCORT_PAIR)

        rid = role_bucket_detailed_id(
            4,
            n_attack=fo,
            n_defend=z,
            n_intercept=torch.tensor([2.0]),
            escort_cnt=z,
            nb_to_ec=o,
            has_blue_carrier=z.bool(),
            has_red_carrier=torch.ones(1, dtype=torch.bool),
            n_alive=fo,
        )
        self.assertEqual(int(rid[0].item()), ROLE4_INTERCEPT_PAIR)

        rid = role_bucket_detailed_id(
            4,
            n_attack=torch.tensor([3.0]),
            n_defend=o,
            n_intercept=z,
            escort_cnt=z,
            nb_to_ec=o,
            has_blue_carrier=z.bool(),
            has_red_carrier=z.bool(),
            n_alive=fo,
        )
        self.assertEqual(int(rid[0].item()), ROLE4_THREE_ONE)

        rid = role_bucket_detailed_id(
            4,
            n_attack=tw,
            n_defend=tw,
            n_intercept=z,
            escort_cnt=z,
            nb_to_ec=o,
            has_blue_carrier=z.bool(),
            has_red_carrier=z.bool(),
            n_alive=fo,
        )
        self.assertEqual(int(rid[0].item()), ROLE4_TWO_TWO)

        rid = role_bucket_detailed_id(
            4,
            n_attack=o,
            n_defend=torch.tensor([3.0]),
            n_intercept=z,
            escort_cnt=z,
            nb_to_ec=o,
            has_blue_carrier=z.bool(),
            has_red_carrier=z.bool(),
            n_alive=fo,
        )
        self.assertEqual(int(rid[0].item()), ROLE4_ONE_THREE)

        rid = role_bucket_detailed_id(
            4,
            n_attack=fo,
            n_defend=z,
            n_intercept=z,
            escort_cnt=z,
            nb_to_ec=o,
            has_blue_carrier=z.bool(),
            has_red_carrier=z.bool(),
            n_alive=fo,
        )
        self.assertEqual(int(rid[0].item()), ROLE4_ALL_PUSH)

        rid = role_bucket_detailed_id(
            4,
            n_attack=z,
            n_defend=fo,
            n_intercept=z,
            escort_cnt=z,
            nb_to_ec=o,
            has_blue_carrier=z.bool(),
            has_red_carrier=z.bool(),
            n_alive=fo,
        )
        self.assertEqual(int(rid[0].item()), ROLE4_TURTLE)

    def test_attack_defense_ratio_buckets_numpy(self) -> None:
        r = np.asarray([0.1, 0.5, 0.9], dtype=np.float64)
        b = attack_defense_ratio_bucket_id(r)
        self.assertListEqual([int(x) for x in b], [0, 1, 2])

    def test_pressure_buckets(self) -> None:
        ip = np.asarray([0.1, 0.5, 0.9], dtype=np.float64)
        dp = np.asarray([0.1, 0.5, 0.9], dtype=np.float64)
        b = pressure_bucket_id(ip, dp)
        self.assertEqual(b.shape, (3,))
        self.assertTrue(all(0 <= int(x) <= 2 for x in b))

    def test_n_telemetry(self) -> None:
        self.assertEqual(N_TELEMETRY, 13)


if __name__ == "__main__":
    unittest.main()
