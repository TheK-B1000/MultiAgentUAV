"""Unit tests for team-phase labels and outcome ids from global state."""

from __future__ import annotations

import unittest

import numpy as np

from rl.discrete_mi import discrete_mi_plugin
from rl.latent_phase_labels import (
    TEAM_PHASES,
    outcome_id_from_global_state,
    team_phase_id_from_global_state,
    team_phase_label_from_global_state,
)


def _state(
    *,
    min_b_rf: float = 0.5,
    min_r_bf: float = 0.5,
    blue_cap: float = 0.0,
    red_cap: float = 0.0,
    b_score: float = 0.0,
    r_score: float = 0.0,
) -> np.ndarray:
    s = np.zeros(19, dtype=np.float32)
    s[8] = min_b_rf
    s[9] = min_r_bf
    s[10] = blue_cap
    s[11] = red_cap
    s[14] = b_score
    s[15] = r_score
    return s


class LatentPhaseLabelsTests(unittest.TestCase):
    def test_stalemate_priority(self) -> None:
        s = _state(blue_cap=1.0, red_cap=0.0)
        pid = team_phase_id_from_global_state(s, stalemate_frac=0.9)
        self.assertEqual(TEAM_PHASES[pid], "stalemate")

    def test_enemy_carrying(self) -> None:
        s = _state(blue_cap=1.0, red_cap=0.0, min_b_rf=0.5, min_r_bf=0.5)
        self.assertEqual(team_phase_label_from_global_state(s, stalemate_frac=0.0), "enemy_carrying_our_flag")

    def test_carrying_home_vs_attack(self) -> None:
        s_far = _state(blue_cap=0.0, red_cap=1.0, min_b_rf=0.5, min_r_bf=0.5)
        self.assertEqual(team_phase_label_from_global_state(s_far, stalemate_frac=0.0), "carrying_flag_home")
        s_near = _state(blue_cap=0.0, red_cap=1.0, min_b_rf=0.1, min_r_bf=0.5)
        self.assertEqual(team_phase_label_from_global_state(s_near, stalemate_frac=0.0), "attacking_enemy_flag")

    def test_defending_own_flag(self) -> None:
        s = _state(min_r_bf=0.1, min_b_rf=0.5)
        self.assertEqual(team_phase_label_from_global_state(s, stalemate_frac=0.0), "defending_own_flag")

    def test_outcome_id(self) -> None:
        self.assertEqual(outcome_id_from_global_state(_state(b_score=0.6, r_score=0.1)), 2)
        self.assertEqual(outcome_id_from_global_state(_state(b_score=0.1, r_score=0.6)), 0)
        self.assertEqual(outcome_id_from_global_state(_state(b_score=0.5, r_score=0.5)), 1)

    def test_mi_non_negative_on_synthetic_joint(self) -> None:
        joint = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]], dtype=np.float64)
        mi = discrete_mi_plugin(joint)
        self.assertGreaterEqual(mi, 0.0)


if __name__ == "__main__":
    unittest.main()
