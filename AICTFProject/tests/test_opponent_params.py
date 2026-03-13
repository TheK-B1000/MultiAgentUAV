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
            self.assertTrue(((params["speed_mult"] >= 0.60) & (params["speed_mult"] <= 1.30)).all().item())
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

        fast_pressure = (params["speed_mult"] > 1.0) & (params["role_switch_prob"] > 0.45)
        slow_counter = (params["speed_mult"] < 0.90) & (params["deception_prob"] > 0.25) & (params["role_switch_prob"] < 0.16)
        steady_balanced = (
            (params["speed_mult"] >= 0.88)
            & (params["speed_mult"] <= 1.04)
            & (params["deception_prob"] >= 0.14)
            & (params["role_switch_prob"] >= 0.18)
            & (params["role_switch_prob"] <= 0.36)
        )
        chaotic = (params["role_switch_prob"] > 0.58) & (params["deception_prob"] > 0.22)

        self.assertTrue(fast_pressure.any().item())
        self.assertTrue(slow_counter.any().item())
        self.assertTrue(steady_balanced.any().item())
        self.assertTrue(chaotic.any().item())


if __name__ == "__main__":
    unittest.main()
