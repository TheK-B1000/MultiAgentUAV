import unittest

import numpy as np

from experiments.run_sppo_d1_localization import _categories_from_vec


class SppoD1LocalizationTests(unittest.TestCase):
    def test_categories_reconstruct_from_stored_observation_vector(self):
        vec = np.zeros((2, 2, 20), dtype=np.float32)
        vec[:, :, 0] = np.asarray([[0.25, 0.50], [0.25, 0.50]])
        vec[:, :, 1] = 0.50
        # Reconstruct home (2,10) for row 0 and stolen position (7,10) for row 1.
        for row, fx in enumerate((2.0, 7.0)):
            vec[row, :, 6] = (fx - vec[row, :, 0] * 19.0) / 20.0
            vec[row, :, 7] = (10.0 - vec[row, :, 1] * 19.0) / 20.0
        vec[1, 0, 10] = 1.0
        cats, audit = _categories_from_vec(vec)
        self.assertEqual(cats["own_flag_home"].tolist(), [True, False])
        self.assertEqual(cats["own_flag_stolen"].tolist(), [False, True])
        self.assertEqual(cats["carrying"].tolist(), [False, True])
        self.assertLess(audit["max_flag_reconstruction_disagreement_between_agents"], 1e-5)


if __name__ == "__main__":
    unittest.main()
