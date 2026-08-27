import unittest

import numpy as np

from experiments.run_sppo_d1_localization import (
    EXPECTED_VEC_DIM,
    _assert_obs_schema,
    _categories_from_vec,
)


def _home_and_stolen_examples() -> np.ndarray:
    """Two rows x two agents x 20 features: row0 home (2,10), row1 stolen (7,10)."""
    vec = np.zeros((2, 2, EXPECTED_VEC_DIM), dtype=np.float32)
    vec[:, :, 0] = np.asarray([[0.25, 0.50], [0.25, 0.50]])
    vec[:, :, 1] = 0.50
    for row, fx in enumerate((2.0, 7.0)):
        vec[row, :, 6] = (fx - vec[row, :, 0] * 19.0) / 20.0
        vec[row, :, 7] = (10.0 - vec[row, :, 1] * 19.0) / 20.0
    vec[1, 0, 10] = 1.0  # carrying on stolen row
    return vec


class SppoD1LocalizationTests(unittest.TestCase):
    def test_categories_reconstruct_home_vs_stolen(self):
        cats, audit = _categories_from_vec(_home_and_stolen_examples())
        self.assertEqual(cats["own_flag_home"].tolist(), [True, False])
        self.assertEqual(cats["own_flag_stolen"].tolist(), [False, True])
        self.assertEqual(cats["carrying"].tolist(), [False, True])
        self.assertEqual(cats["not_carrying"].tolist(), [True, False])
        self.assertEqual(audit["home_x"], 2.0)
        self.assertEqual(audit["home_y"], 10.0)
        self.assertLess(audit["max_flag_reconstruction_disagreement_between_agents"], 1e-5)

    def test_obs_schema_rejects_wrong_vec_dim(self):
        bad = np.zeros((2, 2, 19), dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "vec_dim"):
            _assert_obs_schema(bad)

    def test_obs_schema_rejects_wrong_agent_count(self):
        bad = np.zeros((2, 3, EXPECTED_VEC_DIM), dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "n_agents"):
            _assert_obs_schema(bad)

    def test_flag_reconstruction_disagreement_fails_closed(self):
        vec = _home_and_stolen_examples()
        # Break agent-1 flag reconstruction on row 0.
        vec[0, 1, 6] = 0.9
        with self.assertRaisesRegex(RuntimeError, "flag-reconstruction disagreement"):
            _categories_from_vec(vec)


if __name__ == "__main__":
    unittest.main()
