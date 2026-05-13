"""Unit tests for plug-in MI on discrete joint counts."""

from __future__ import annotations

import unittest

import numpy as np

from rl.discrete_mi import discrete_mi_plugin


class DiscreteMiTests(unittest.TestCase):
    def test_uniform_independent_is_near_zero(self) -> None:
        counts = np.ones((3, 4), dtype=np.float64)
        self.assertLess(discrete_mi_plugin(counts), 1e-6)

    def test_empty_counts(self) -> None:
        self.assertEqual(discrete_mi_plugin(np.zeros((2, 2))), 0.0)

    def test_one_hot_joint_has_positive_mi(self) -> None:
        counts = np.eye(4, dtype=np.float64)
        self.assertGreater(discrete_mi_plugin(counts), 0.2)

    def test_requires_two_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            discrete_mi_plugin(np.ones(5))


if __name__ == "__main__":
    unittest.main()
