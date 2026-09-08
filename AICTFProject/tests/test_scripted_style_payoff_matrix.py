from __future__ import annotations

import unittest

from experiments.run_scripted_style_payoff_matrix import (
    DEFAULT_MAPS,
    NICHE_CANONICAL_MAP,
    _analysis_rows,
    _episode_seed,
)


class ScriptedStylePayoffMatrixTests(unittest.TestCase):
    def test_niche_canonical_map_default_is_map_a(self) -> None:
        self.assertEqual(NICHE_CANONICAL_MAP, "map_a")
        self.assertEqual(DEFAULT_MAPS, ("map_a",))

    def test_episode_seed_is_independent_of_blue_style(self) -> None:
        seed_a = _episode_seed(260726, red_index=2, map_index=1, episode_index=7)
        seed_b = _episode_seed(260726, red_index=2, map_index=1, episode_index=7)
        self.assertEqual(seed_a, seed_b)

        self.assertNotEqual(seed_a, _episode_seed(260726, red_index=3, map_index=1, episode_index=7))
        self.assertNotEqual(seed_a, _episode_seed(260726, red_index=2, map_index=0, episode_index=7))
        self.assertNotEqual(seed_a, _episode_seed(260726, red_index=2, map_index=1, episode_index=8))

    def test_analysis_rows_use_red_map_as_column(self) -> None:
        rows = [
            {
                "blue_style": "BLUE_RUSH",
                "red_style": "OP11_ADAPTIVE_EXPLOITER",
                "map": "map_b_split_lane",
                "episode_index": 0,
                "win_margin": 1,
            }
        ]
        out = _analysis_rows(rows)
        self.assertEqual(out[0]["red_style"], "OP11_ADAPTIVE_EXPLOITER|map_b_split_lane")


if __name__ == "__main__":
    unittest.main()
