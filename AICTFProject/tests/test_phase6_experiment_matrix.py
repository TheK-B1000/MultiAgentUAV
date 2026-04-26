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
    def test_must_have_baselines_are_named_explicitly(self) -> None:
        names = {variant.name for variant in VARIANTS}
        self.assertIn("flat_ppo_marl", names)
        self.assertIn("latent_no_persistence", names)
        self.assertIn("fixed_latent", names)

        self.assertEqual(_variant("flat_ppo_marl").train_flags, ("--no-latent-strategy",))
        self.assertEqual(
            _variant("latent_no_persistence").train_flags,
            ("--latent-resample-every", "20", "--latent-lam-p", "0.0"),
        )
        self.assertEqual(
            _variant("fixed_latent").train_flags,
            ("--fixed-latent-strategy", "--fixed-latent-id", "0"),
        )

    def test_k_ablation_matches_trainer_validation(self) -> None:
        flags = [flag for variant in VARIANTS for flag in variant.train_flags]
        self.assertNotIn("2", _variant("k6").train_flags)
        self.assertNotIn("k2", {variant.name for variant in VARIANTS})
        self.assertNotEqual(flags[flags.index("--latent-k") + 1], "2")
        self.assertEqual(_variant("k6").train_flags, ("--latent-k", "6"))


if __name__ == "__main__":
    unittest.main()
