import unittest

import numpy as np

from experiments.report_sppo_d0_seed_uncertainty import (
    _weighted_cutoff,
    seed_bootstrap,
)


class SppoD0SeedUncertaintyTests(unittest.TestCase):
    def test_weighted_cutoff_uses_seed_multiplicity(self):
        values = np.asarray([-4.0, -1.0, 2.0, 5.0])
        weights = np.asarray([0.0, 3.0, 1.0, 0.0])
        self.assertEqual(_weighted_cutoff(values, weights, 0.25), -1.0)

    def test_seed_bootstrap_is_deterministic_and_reports_dynamic_quartile(self):
        rows = []
        for seed, vals in ((1, (-2.0, -1.0)), (2, (0.0, 1.0)),
                           (3, (2.0, 3.0)), (4, (4.0, 5.0))):
            for idx, margin in enumerate(vals):
                rows.append({
                    "seed": seed,
                    "margin_B_bits": margin,
                    "delta_B_hat_qpsi": margin + 0.25,
                    "qpsi_ranks_z1_correct": float(margin >= 0),
                    "blue_carrying": idx == 0,
                    "own_flag_home": seed % 2 == 0,
                    "tertile": "early" if idx == 0 else "late",
                })
        a, qa = seed_bootstrap(rows, samples=200, alpha=0.05, rng_seed=7)
        b, qb = seed_bootstrap(rows, samples=200, alpha=0.05, rng_seed=7)
        self.assertEqual(a, b)
        self.assertEqual(qa, qb)
        self.assertIn("worst_quartile_margin_B", a)
        self.assertLess(qa["bootstrap_cutoff_lcb95"], qa["bootstrap_cutoff_ucb95"])
        self.assertEqual(
            a["ALL"]["mean_margin_B_bits"]["valid_bootstrap_samples"], 200
        )


if __name__ == "__main__":
    unittest.main()
