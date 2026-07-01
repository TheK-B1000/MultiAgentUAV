"""OP5_RUSHER scripted opponent produces bounded, finite GPU tensors."""

from __future__ import annotations

import unittest

import torch

from opponent_params import sample_batched_opponent_params


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


class Op5RusherParamsTests(unittest.TestCase):
    def test_op5_rusher_bounded_2v2(self) -> None:
        p = sample_batched_opponent_params(
            kind="SCRIPTED",
            key="OP5_RUSHER",
            phase="OP5_RUSHER",
            n_agents=2,
            batch_size=256,
            device="cpu",
            generator=_gen(202),
        )
        self.assertTrue(torch.isfinite(p["speed_mult"]).all().item())
        self.assertTrue(((p["speed_mult"] >= 0.60) & (p["speed_mult"] <= 1.45)).all().item())
        self.assertGreater(float(p["speed_mult"].mean().item()), 1.12)

    def test_op5_alias_op5(self) -> None:
        a = sample_batched_opponent_params(
            "SCRIPTED", "OP5", "OP5", n_agents=4, batch_size=8, device="cpu", generator=_gen(3)
        )
        b = sample_batched_opponent_params(
            "SCRIPTED", "OP5_RUSHER", "OP5_RUSHER", n_agents=4, batch_size=8, device="cpu", generator=_gen(3)
        )
        self.assertTrue(torch.allclose(a["speed_mult"], b["speed_mult"]))


if __name__ == "__main__":
    unittest.main()
