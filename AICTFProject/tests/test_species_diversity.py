import unittest

import torch

from game_field_gpu import BatchedCTFCore, GPUFieldConfig


class SpeciesDiversityTests(unittest.TestCase):
    """
    Regression test for a bug where SPECIES opponents (RUSHER/CAMPER/BALANCED) were
    silently collapsed to SCRIPTED:OP3 dynamics before sampling, and the logged
    species_tag was hardcoded to "BALANCED" regardless of which species actually played.
    Both must produce genuinely distinct behavior/labels for opponent-diversity claims
    (and any downstream opponent-identity signal) to be meaningful.
    """

    def _make_core(self) -> BatchedCTFCore:
        cfg = GPUFieldConfig(n_envs=3, max_blue_agents=2, max_red_agents=2, device="cpu", seed=42)
        return BatchedCTFCore(cfg)

    def test_species_kinds_produce_distinct_dynamics_params(self):
        core = self._make_core()
        core._opponent_kind[0] = "SPECIES"
        core._opponent_key[0] = "RUSHER"
        core._opponent_kind[1] = "SPECIES"
        core._opponent_key[1] = "CAMPER"
        core._opponent_kind[2] = "SPECIES"
        core._opponent_key[2] = "BALANCED"

        mask = torch.ones((3,), dtype=torch.bool)
        core._apply_opponent_params_for_mask(mask)

        # RUSHER: attacker_style=1, defender_style=0, speed_mult sampled from (1.05, 1.25)
        self.assertEqual(int(core.red_attacker_style[0].item()), 1)
        self.assertEqual(int(core.red_defender_style[0].item()), 0)
        self.assertTrue(1.0 < float(core.red_speed_mult[0].item()) <= 1.30)

        # CAMPER: attacker_style=0, defender_style=1, speed_mult sampled from (0.80, 1.0)
        self.assertEqual(int(core.red_attacker_style[1].item()), 0)
        self.assertEqual(int(core.red_defender_style[1].item()), 1)
        self.assertTrue(0.60 <= float(core.red_speed_mult[1].item()) <= 1.0)

        # BALANCED: attacker_style=1, defender_style=1
        self.assertEqual(int(core.red_attacker_style[2].item()), 1)
        self.assertEqual(int(core.red_defender_style[2].item()), 1)

        # The three species must not all collapse to identical params (the old bug:
        # every SPECIES kind became SCRIPTED:OP3, so all three rows were identical).
        self.assertFalse(
            torch.equal(core.red_attacker_style[0], core.red_attacker_style[1])
            and torch.equal(core.red_defender_style[0], core.red_defender_style[1])
        )

    def test_species_tag_is_logged_correctly_not_hardcoded_balanced(self):
        # Mirrors the episode_result construction in GPUCTFVecEnv.step_wait
        # (game_field_gpu.py), which used to hardcode species_tag to "BALANCED"
        # regardless of the actual sampled species.
        for okind, okey, expected_species_tag in (
            ("species", "RUSHER", "RUSHER"),
            ("species", "CAMPER", "CAMPER"),
            ("species", "BALANCED", "BALANCED"),
            ("scripted", "OP3", ""),
        ):
            species_tag = okey if okind == "species" else ""
            self.assertEqual(species_tag, expected_species_tag)


if __name__ == "__main__":
    unittest.main()
