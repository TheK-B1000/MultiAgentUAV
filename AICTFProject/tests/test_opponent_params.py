import unittest

import torch

from opponent_params import sample_batched_opponent_params


def _make_generator(seed: int) -> torch.Generator:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return gen


class OpponentParamsTests(unittest.TestCase):
    def test_op4_outputs_are_bounded_and_finite_across_team_sizes(self):
        for n_agents in (2, 4, 8):
            params = sample_batched_opponent_params(
                kind="SCRIPTED",
                key="OP4",
                phase="OP4",
                n_agents=n_agents,
                batch_size=512,
                device="cpu",
                generator=_make_generator(100 + n_agents),
            )

            self.assertEqual(params["speed_mult"].shape, (512,))
            self.assertEqual(params["deception_prob"].shape, (512,))
            self.assertEqual(params["attack_sync_window"].shape, (512,))
            self.assertEqual(params["coordinated_attack"].dtype, torch.bool)
            self.assertTrue(torch.isfinite(params["speed_mult"]).all().item())
            self.assertTrue(torch.isfinite(params["deception_prob"]).all().item())
            self.assertTrue(torch.isfinite(params["noise_sigma"]).all().item())
            self.assertTrue(((params["speed_mult"] >= 0.60) & (params["speed_mult"] <= 1.45)).all().item())
            self.assertTrue(((params["deception_prob"] >= 0.0) & (params["deception_prob"] <= 0.60)).all().item())
            self.assertTrue(((params["role_switch_prob"] >= 0.0) & (params["role_switch_prob"] <= 0.90)).all().item())
            self.assertTrue(((params["noise_sigma"] >= 0.0) & (params["noise_sigma"] <= 0.10)).all().item())
            self.assertTrue(((params["attack_sync_window"] >= 0) & (params["attack_sync_window"] <= 8)).all().item())
            self.assertTrue(((params["attacker_style"] == 0) | (params["attacker_style"] == 1)).all().item())
            self.assertTrue(((params["defender_style"] == 0) | (params["defender_style"] == 1)).all().item())

    def test_op4_sampling_is_reproducible_with_fixed_seed(self):
        params_a = sample_batched_opponent_params(
            kind="SCRIPTED",
            key="OP4",
            phase="OP4",
            n_agents=2,
            batch_size=128,
            device="cpu",
            generator=_make_generator(7),
        )
        params_b = sample_batched_opponent_params(
            kind="SCRIPTED",
            key="OP4",
            phase="OP4",
            n_agents=2,
            batch_size=128,
            device="cpu",
            generator=_make_generator(7),
        )

        for key in params_a:
            self.assertTrue(torch.equal(params_a[key], params_b[key]), key)

    def test_op4_samples_multiple_behavior_regimes(self):
        params = sample_batched_opponent_params(
            kind="SCRIPTED",
            key="OP4",
            phase="OP4",
            n_agents=2,
            batch_size=2048,
            device="cpu",
            generator=_make_generator(19),
        )

        # OP4 is a held-out style mixture (blitz / anchor-trap / volatile pivot / yolo), not OP3 + ε.
        committed_blitz = (
            (params["speed_mult"] > 1.02)
            & (params["role_switch_prob"] < 0.23)
            & (params["attacker_style"] == 1)
            & (params["defender_style"] == 0)
        )
        anchor_trap = (params["speed_mult"] < 0.88) & (params["deception_prob"] > 0.26)
        volatile_dual = (
            (params["attacker_style"] == 1)
            & (params["defender_style"] == 1)
            & (params["role_switch_prob"] > 0.50)
        )
        yolo = (params["speed_mult"] > 1.00) & (params["role_switch_prob"] > 0.60)

        self.assertTrue(committed_blitz.any().item())
        self.assertTrue(anchor_trap.any().item())
        self.assertTrue(volatile_dual.any().item())
        self.assertTrue(yolo.any().item())

    def test_op6_outputs_bounded_and_style_for_4v4(self) -> None:
        params = sample_batched_opponent_params(
            kind="SCRIPTED",
            key="OP6",
            phase="OP6",
            n_agents=4,
            batch_size=64,
            device="cpu",
            generator=_make_generator(501),
        )
        self.assertTrue(torch.isfinite(params["speed_mult"]).all().item())
        self.assertTrue((params["attacker_style"] == 0).all().item())
        self.assertTrue((params["defender_style"] == 1).all().item())
        self.assertTrue(((params["role_switch_prob"] >= 0.0) & (params["role_switch_prob"] <= 0.90)).all().item())

    def test_op7_samples_multiple_regimes_like_mixture(self) -> None:
        params = sample_batched_opponent_params(
            kind="SCRIPTED",
            key="OP7",
            phase="OP7",
            n_agents=4,
            batch_size=4096,
            device="cpu",
            generator=_make_generator(701),
        )
        shell = (params["speed_mult"] < 0.86) & (params["attacker_style"] == 0) & (params["defender_style"] == 1)
        feint = (params["attacker_style"] == 1) & (params["defender_style"] == 0) & (params["role_switch_prob"] > 0.38)
        dual = (params["attacker_style"] == 1) & (params["defender_style"] == 1) & (params["role_switch_prob"] > 0.46)
        surge = (params["speed_mult"] > 0.91) & (params["attacker_style"] == 1) & (params["defender_style"] == 0)
        self.assertTrue(shell.any().item())
        self.assertTrue(feint.any().item())
        self.assertTrue(dual.any().item())
        self.assertTrue(surge.any().item())


if __name__ == "__main__":
    unittest.main()
