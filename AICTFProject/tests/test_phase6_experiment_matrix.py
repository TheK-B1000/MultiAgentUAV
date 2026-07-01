"""Guards for final experiment command generation."""

from __future__ import annotations

import unittest

from experiments.phase6_experiment_matrix import VARIANTS


def _variant(name: str):
    for variant in VARIANTS:
        if variant.name == name:
            return variant
    raise AssertionError(f"missing phase6 variant {name!r}")


class Phase6ExperimentMatrixTests(unittest.TestCase):
    def test_professor_requested_baselines_are_named_explicitly(self) -> None:
        names = {variant.name for variant in VARIANTS}
        self.assertIn("curriculum", names)
        self.assertIn("no_latent", names)

        self.assertEqual(_variant("latent_default").mode, "FIXED_OPPONENT")
        self.assertEqual(_variant("latent_default").fixed_opponent, "OP3")
        self.assertEqual(_variant("latent_default").train_flags, ())
        self.assertEqual(_variant("curriculum").mode, "CURRICULUM")
        self.assertEqual(_variant("curriculum").train_flags, ("--no-latent-strategy",))
        self.assertEqual(_variant("no_latent").mode, "FIXED_OPPONENT")
        self.assertEqual(_variant("no_latent").fixed_opponent, "OP3")
        self.assertEqual(_variant("no_latent").train_flags, ("--no-latent-strategy",))

    def test_removed_extra_ablations_stay_out_of_final_matrix(self) -> None:
        flags = [flag for variant in VARIANTS for flag in variant.train_flags]
        names = {variant.name for variant in VARIANTS}
        self.assertNotIn("fixed_latent", names)
        self.assertNotIn("k6", names)
        self.assertNotIn("sparse20", names)
        self.assertNotIn("latent_no_persistence", names)
        self.assertNotIn("--latent-k", flags)


if __name__ == "__main__":
    unittest.main()
